import argparse
import time
from collections import defaultdict
import shutil

from pathlib import Path

import wandb
import tqdm
import numpy as np
import torch
import torchvision
import yaml
import pandas as pd
import plotly.graph_objects as go

from PIL import Image, ImageDraw, ImageFont

from model import ConditionalStateEncoderImitation
from habitat_dataset import get_dataset, HabitatDataset
from habitat_wrapper import TASKS, MODELS, Rollout, get_episode, save_episode

all_success = []
all_spl = []
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 20)]

NUM_EPISODES = 25
VIDEO_FREQ   = 5
EPOCH_FREQ   = 5

def _get_box(all_x):
    fig = go.Figure(data=[go.Box(y=data,
        boxpoints='all',
        boxmean=True,
        jitter=0.1,
        pointpos=-1.6,
        name=f"{max(wandb.run.summary['epoch']-20, 0)+(EPOCH_FREQ*i)}",
        marker_color=c[i]
    ) for i, data in enumerate(all_x[-20:])])
    fig.update_layout(
        xaxis=dict(title='Epoch', showgrid=False, zeroline=False, dtick=1),
        yaxis=dict(zeroline=False, gridcolor='white'),
        paper_bgcolor='rgb(233,233,233)',
        plot_bgcolor='rgb(233,233,233)',
        showlegend=False
    )

    return fig

def _get_hist2d(x, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        showlegend=False,
        marker=dict(
            symbol='x',
            opacity=0.7,
            color='white',
            size=8,
            line=dict(width=1),
        )
    ))

    fig.add_trace(go.Histogram2d(
        x=x,
        y=y,
        autobinx=False,
        xbins=dict(start=0, end=25, size=2),
        autobiny=False,
        ybins=dict(start=0, end=1, size=0.05),
        histnorm='probability density',
        colorscale=["#cc0000", "#4e9a06", "#73d216", "#8ae234"]
    ))

    fig.update_layout(
        xaxis=dict( ticks='', showgrid=False, zeroline=False, range=[0, 25] ),
        yaxis=dict( ticks='', showgrid=False, zeroline=False, nticks=20, range=[0,1] ),
        autosize=False,
        height=550,
        width=550,
        hovermode='closest',
    )

    return fig


def validate(net, env, data, config):
    net.eval()
    net.batch_size = config['data_args']['batch_size']
    env.mode = 'student'

    losses = list()
    criterion = torch.nn.CrossEntropyLoss()
    tick = time.time()

    # static validation set
    for i, (rgb, action, prev_action, meta, mask) in enumerate(tqdm.tqdm(data, desc='val', total=len(data), leave=False)):
        net.clean()

        rgb = rgb.to(config['device'])
        action = action.to(config['device'])
        prev_action = prev_action.to(config['device'])
        meta = meta.to(config['device'])
        mask = mask.to(config['device'])

        episode_loss = 0
        for t in range(rgb.shape[0]):
            _action = net((rgb[t], meta[t], prev_action[t], mask[t]))

            loss = criterion(_action, action[t])
            episode_loss += loss.mean()

        losses.append(episode_loss.mean().item())

        metrics = {
            'loss': episode_loss.mean().item(),
            'images_per_second': (rgb.shape[0]*rgb.shape[1]) / (time.time() - tick)
        }

        wandb.log(
                {('%s/%s' % ('val', k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    # rollout
    net.batch_size = 1
    if wandb.run.summary['epoch'] % EPOCH_FREQ == 0:
        distance_to_goal = np.zeros(NUM_EPISODES)
        success = np.zeros(NUM_EPISODES)
        spl = np.zeros(NUM_EPISODES)
        distance_from_goal = np.zeros(NUM_EPISODES)

        for ep in range(NUM_EPISODES):
            images = []

            net.clean()
            for step in get_episode(env):
                if ep % VIDEO_FREQ == 0:
                    frame = Image.fromarray(step['rgb'])
                    draw = ImageDraw.Draw(frame)
                    font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf', 18)
                    direction = env.get_direction()
                    draw.text((0, 0), '({: <5.1f}, {: <5.1f}) {: <4.1f}'.format(*direction, np.linalg.norm(direction)), (0, 0, 0), font=font)

                    images.append(np.transpose(np.uint8(frame), (2, 0, 1)))

            env_metrics = env.env.get_metrics()
            distance_to_goal[ep] = env_metrics['distance_to_goal']
            success[ep] = env_metrics['success']
            spl[ep] = env_metrics['spl']
            distance_from_goal[ep] = np.linalg.norm(env.env.current_episode.goals[0].position[:2] - env.state.position[:2])

            metrics = {}
            if ep == NUM_EPISODES - 1:
                all_success.append(success)
                metrics['success_mean'] = np.mean(success)
                metrics['success_std'] = np.std(success)
                metrics['success_median'] = np.median(success)
                metrics['success'] = wandb.Histogram(success)
                metrics['success_box'] = _get_box(all_success)

                all_spl.append(spl)
                metrics['spl_mean'] = np.mean(spl)
                metrics['spl_std'] = np.std(spl)
                metrics['spl_median'] = np.median(spl)
                metrics['spl'] = wandb.Histogram(spl)
                metrics['spl_box'] = _get_box(all_spl)

                metrics['dtg_mean'] = np.mean(distance_to_goal)
                metrics['dfg_mean'] = np.mean(distance_from_goal)

                metrics['distance_to_goal_vs_success'] = _get_hist2d(distance_to_goal, success)
                metrics['distance_to_goal_vs_spl'] = _get_hist2d(distance_to_goal, spl)
                metrics['distance_from_goal_vs_success'] = _get_hist2d(distance_from_goal, success)

            if ep % VIDEO_FREQ == 0 and len(images) > 0:
                metrics[f'video_{(ep//VIDEO_FREQ)+1}'] = wandb.Video(np.array(images), fps=20, format='mp4')

            wandb.log(
                    {('%s/%s' % ('val', k)): v for k, v in metrics.items()},
                    step=wandb.run.summary['step'])

    return np.mean(losses)


def train(net, env, data, optim, config):
    net.train()
    net.batch_size = config['data_args']['batch_size']
    env.mode = 'teacher'

    losses = list()
    criterion = torch.nn.CrossEntropyLoss()
    tick = time.time()

    for i, (rgb, action, prev_action, meta, mask) in enumerate(tqdm.tqdm(data, desc='train', total=len(data), leave=False)):
        # rgb.shape
        # sequence, batch, ...

        net.clean()

        rgb = rgb.to(config['device'])
        action = action.to(config['device'])
        prev_action = prev_action.to(config['device'])
        meta = meta.to(config['device'])
        mask = mask.to(config['device'])

        episode_loss = 0
        for t in range(rgb.shape[0]):
            _action = net((rgb[t], meta[t], prev_action[t], mask[t]))

            loss = criterion(_action, action[t])
            loss.backward()

            episode_loss += loss.mean()

            optim.step()
            optim.zero_grad()

        losses.append(episode_loss.mean().item())

        wandb.run.summary['step'] += 1

        metrics = {
            'loss': episode_loss.mean().item(),
            'images_per_second': (rgb.shape[0]*rgb.shape[1]) / (time.time() - tick)
        }

        wandb.log(
                {('%s/%s' % ('train', k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    return np.mean(losses)


def resume_project(net, optim, scheduler, config):
    print('Resumed at epoch %d.' % wandb.run.summary['epoch'])

    net.load_state_dict(torch.load(config['checkpoint_dir'] / 'model_latest.t7'))
    optim.load_state_dict(torch.load(config['checkpoint_dir'] / 'optim_latest.t7'))
    scheduler.load_state_dict(torch.load(config['checkpoint_dir'] / 'scheduler_latest.t7'))


def checkpoint_project(net, optim, scheduler, config):
    torch.save(net.state_dict(), config['checkpoint_dir'] / 'model_latest.t7')
    torch.save(optim.state_dict(), config['checkpoint_dir'] / 'optim_latest.t7')
    torch.save(scheduler.state_dict(), config['checkpoint_dir'] / 'scheduler_latest.t7')


def main(config):
    net = ConditionalStateEncoderImitation(config['data_args']['batch_size'], **config['student_args']).to(config['device'])
    data_train, data_val = get_dataset(**config['data_args'])

    #env_train = Rollout(**config['teacher_args'], student=net, rnn=True, split='train')
    env_val = Rollout(**config['teacher_args'], student=net, rnn=True, split='val')

    optim = torch.optim.Adam(net.parameters(), **config['optimizer_args'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim,
            milestones=[config['max_epoch'] * 0.5, config['max_epoch'] * 0.75],
            gamma=0.5)

    project_name = 'habitat-{}-{}-student'.format(
            config['teacher_args']['task'], config['teacher_args']['proxy'])
    wandb.init(project=project_name, config=config, id=config['run_name'], resume='auto')
    wandb.save(str(Path(wandb.run.dir) / '*.t7'))

    if wandb.run.resumed:
        resume_project(net, optim, scheduler, config)
    else:
        wandb.run.summary['step'] = 0
        wandb.run.summary['epoch'] = 0

    for epoch in tqdm.tqdm(range(wandb.run.summary['epoch'], config['max_epoch']+1), desc='epoch'):
        wandb.run.summary['epoch'] = epoch

        checkpoint_project(net, optim, scheduler, config)

        loss_train = train(net, env_val, data_train, optim, config)
        with torch.no_grad():
            loss_val = validate(net, env_val, data_val, config)

        scheduler.step()

        wandb.log(
                {'train/loss_epoch': loss_train, 'val/loss_epoch': loss_val},
                step=wandb.run.summary['step'])

        if loss_val < wandb.run.summary.get('best_val_loss', np.inf):
            wandb.run.summary['best_val_loss'] = loss_val
            wandb.run.summary['best_epoch'] = epoch

        if epoch % 10 == 0:
            torch.save(net.state_dict(), Path(wandb.run.dir) / ('model_%03d.t7' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=Path, default='checkpoints')

    # Teacher args.
    parser.add_argument('--teacher_task', choices=TASKS, required=True)
    parser.add_argument('--teacher_proxy', choices=MODELS.keys(), required=True)

    # Student args.
    parser.add_argument('--resnet_model', choices=['resnet18', 'resnet50', 'resneXt50', 'se_resnet50', 'se_resneXt101', 'se_resneXt50'])
    parser.add_argument('--conditional', action='store_true')

    # Data args.
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--dataset_size', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--interpolate', action='store_true')
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--capacity', type=int, default=1000)          # if DAgger
    parser.add_argument('--episodes_per_epoch', type=int, default=100) # if DAgger

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()

    run_name = '-'.join(map(str, [
        parsed.resnet_model,
        'conditional' if parsed.conditional else 'direct', 'bc',         # run-specific, high-level
        'aug' if parsed.augmentation else 'noaug', 'interpolate' if parsed.interpolate else 'original', # dataset
        #*((parsed.episodes_per_epoch, parsed.capacity) if parsed.dagger else ()),                       # DAgger
        parsed.dataset_size, parsed.batch_size, parsed.lr, parsed.weight_decay                          # boring stuff
    ])) + '-v11.11'

    checkpoint_dir = parsed.checkpoint_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
            'run_name': run_name,
            'notes': 'rnn setup from ddppo paper',
            'max_epoch': parsed.max_epoch,
            'checkpoint_dir': checkpoint_dir,

            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

            'teacher_args': {
                'task': parsed.teacher_task,
                'proxy': parsed.teacher_proxy
                },

            'student_args': {
                'resnet_model': parsed.resnet_model
                },

            'data_args': {
                'dataset_dir': parsed.dataset_dir,
                'dataset_size': parsed.dataset_size,
                'batch_size': parsed.batch_size,
                'rnn': True,

                'interpolate': parsed.interpolate,
                'augmentation': parsed.augmentation,

                'capacity': parsed.capacity,
                'episodes_per_epoch': parsed.episodes_per_epoch,
                },

            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay
                }
            }

    main(config)
