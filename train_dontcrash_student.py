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

from PIL import Image, ImageDraw

from model import get_model
from habitat_dataset import get_dataset, HabitatDataset
from habitat_wrapper import TASKS, MODELS, get_episode, save_episode

all_lwns = []
all_lwns_norm = []
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 20)]

def validate(net, env, data, config):
    NUM_EPISODES = 100
    VIDEO_FREQ   = 20
    EPOCH_FREQ   = 5

    net.eval()
    env.mode = 'student'

    losses = list()
    criterion = torch.nn.CrossEntropyLoss()
    tick = time.time()

    # static validation set
    for i, (rgb, _, _, action, _, _) in enumerate(tqdm.tqdm(data, desc='val', total=len(data), leave=False)):
        rgb = rgb.to(config['device'])
        action = torch.LongTensor(action).to(config['device'])

        if config['student_args']['conditional']:
            _action = net((rgb, meta))
        else:
            _action = net((rgb,))

        loss = criterion(_action, action)

        loss_mean = loss.mean()
        losses.append(loss_mean.item())

        metrics = {
            'loss': loss_mean.item(),
            'images_per_second': rgb.shape[0] / (time.time() - tick)
        }

        tick = time.time()

        wandb.log(
                {('%s/%s' % ('val', k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

    # rollout
    if wandb.run.summary['epoch'] % EPOCH_FREQ == 0:
        lwns      = np.zeros(NUM_EPISODES)
        lwns_norm = np.zeros(NUM_EPISODES)
        for ep in range(NUM_EPISODES):
            images = []
            longest, length = 0, 0
            for step in get_episode(env):
                lwns[ep] = max(lwns[ep], longest)
                if step['is_stuck']:
                    longest = 0
                longest += 1
                length += 1

                if ep % VIDEO_FREQ == 0:
                    images.append(np.transpose(step['rgb'], (2, 0, 1)))

            lwns_norm[ep] = lwns[ep] / length

            metrics = {}
            if ep == NUM_EPISODES - 1 and config['teacher_args']['task'] == 'dontcrash':
                all_lwns.append(lwns)
                metrics['lwns_mean'] = np.mean(lwns)
                metrics['lwns_std'] = np.std(lwns)
                metrics['lwns_median'] = np.median(lwns)
                metrics['lwns'] = wandb.Histogram(lwns)
                fig = go.Figure(data=[go.Box(y=data,
                    boxpoints='all',
                    boxmean=True,
                    jitter=0.1,
                    pointpos=-1.6,
                    name=f"{max(wandb.run.summary['epoch']-20, 0)+(EPOCH_FREQ*i)}",
                    marker_color=c[i]
                ) for i, data in enumerate(all_lwns[-20:])])
                fig.update_layout(
                    xaxis=dict(title='Epoch', showgrid=False, zeroline=False, dtick=1),
                    yaxis=dict(zeroline=False, gridcolor='white'),
                    paper_bgcolor='rgb(233,233,233)',
                    plot_bgcolor='rgb(233,233,233)',
                    showlegend=False
                )
                metrics['lwns_box'] = fig

                all_lwns_norm.append(lwns_norm)
                metrics['lwns_norm_mean'] = np.mean(lwns_norm)
                metrics['lwns_norm_std'] = np.std(lwns_norm)
                metrics['lwns_norm_median'] = np.median(lwns_norm)
                metrics['lwns_norm'] = wandb.Histogram(lwns_norm)
                fig = go.Figure(data=[go.Box(y=data,
                    boxpoints='all',
                    boxmean=True,
                    jitter=0.1,
                    pointpos=-1.6,
                    name=f"{max(wandb.run.summary['epoch']-20, 0)+(EPOCH_FREQ*i)}",
                    marker_color=c[i]
                ) for i, data in enumerate(all_lwns_norm[-20:])])
                fig.update_layout(
                    xaxis=dict(title='Epoch', showgrid=False, zeroline=False, dtick=1),
                    yaxis=dict(zeroline=False, gridcolor='white', range=[0., 1.]),
                    paper_bgcolor='rgb(233,233,233)',
                    plot_bgcolor='rgb(233,233,233)',
                    showlegend=False
                )
                metrics['lwns_norm_box'] = fig

            if ep % VIDEO_FREQ == 0 and len(images) > 0:
                metrics[f'video_{(ep//VIDEO_FREQ)+1}'] = wandb.Video(np.array(images), fps=30, format='mp4')

            wandb.log(
                    {('%s/%s' % ('val', k)): v for k, v in metrics.items()},
                    step=wandb.run.summary['step'])

    return np.mean(losses)


def train(net, env, data, optim, config):
    if config['dagger']:
        net.eval()
        env.mode = 'both'

        # allow longer episodes as training progresses
        #env.max_episode_length = (wandb.run.summary['epoch'] + 1) * 20

        # rollout some datasets; aggregate
        summary_file = config['data_args']['dagger_dataset_dir'] / 'summary.csv'
        summary = defaultdict(float)
        summary['ep'] = 1
        if summary_file.exists():
            summary = pd.read_csv(summary_file).iloc[0]

        num_episodes = 0
        while num_episodes < config['data_args']['episodes_per_epoch']:
            episode_dir = config['data_args']['dagger_dataset_dir'] / 'train' / '{:06}'.format(int(summary['ep']) + num_episodes)
            if episode_dir.exists():
                shutil.rmtree(episode_dir, ignore_errors=True)

            episode_dir.mkdir(parents=True, exist_ok=True)
            save_episode(env, episode_dir)
            data.add_episode(HabitatDataset(episode_dir))

            num_episodes += 1

        summary['ep'] += num_episodes
        pd.DataFrame([summary]).to_csv(summary_file, index=False)

        for i, episode in enumerate(data.episodes.datasets):
            episode.episode_idx = i

        # cleanup after dagger
        data.post_dagger()

        episode_loss = np.zeros(len(data.episodes.datasets))
        episode_step = np.ones(len(data.episodes.datasets)) # prevent divide by zero

    losses = list()
    criterion = torch.nn.CrossEntropyLoss()
    tick = time.time()

    net.train()
    env.mode = 'teacher'
    for i, (rgb, _, _, action, _, episode_idx) in enumerate(tqdm.tqdm(data, desc='train', total=len(data), leave=False)):
        rgb = rgb.to(config['device'])
        action = torch.LongTensor(action).to(config['device'])

        if config['student_args']['conditional']:
            _action = net((rgb, meta))
        else:
            _action = net((rgb,))

        loss = criterion(_action, action)
        if config['dagger']:
            episode_loss[episode_idx] += loss.detach().cpu().numpy() # unnormalized
            episode_step[episode_idx] += 1

        loss_mean = loss.mean()
        losses.append(loss_mean.item())

        loss_mean.backward()
        optim.step()
        optim.zero_grad()

        wandb.run.summary['step'] += 1

        metrics = {
            'loss': loss_mean.item(),
            'images_per_second': rgb.shape[0] / (time.time() - tick)
        }

        wandb.log(
                {('%s/%s' % ('train', k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    if config['dagger']:
        # normalize episode losses
        normalized_episode_loss = np.divide(episode_loss, episode_step)
        for i, episode in enumerate(data.episodes.datasets):
            episode.loss = normalized_episode_loss[i]

        # balance heap
        data.post_train()

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
    net = get_model(**config['student_args']).to(config['device'])
    data_train, data_val = get_dataset(**config['data_args'])

    if config['dagger']:
        (config['data_args']['dagger_dataset_dir'] / 'train').mkdir(parents=True, exist_ok=True)

    env = Rollout(**config['teacher_args'], student=net)

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

        loss_train = train(net, env, data_train, optim, config)
        with torch.no_grad():
            loss_val = validate(net, env, data_val, config)

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
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--capacity', type=int, default=1000)         # if DAgger
    parser.add_argument('--episodes_per_epoch', type=int, default=50) # if DAgger
    parser.add_argument('--dagger_dataset_dir', type=Path)            # if DAgger

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()

    run_name = '-'.join(map(str, [
        parsed.resnet_model,
        'conditional' if parsed.conditional else 'direct', 'dagger' if parsed.dagger else 'bc', # run-specific, high-level
        'interpolate' if parsed.interpolate else 'original',                                    # dataset
        *((parsed.episodes_per_epoch, parsed.capacity) if parsed.dagger else ()),               # DAgger
        parsed.dataset_size, parsed.batch_size, parsed.lr, parsed.weight_decay                  # boring stuff
    ])) + '-v8.1'

    checkpoint_dir = parsed.checkpoint_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
            'run_name': run_name,
            'max_epoch': parsed.max_epoch,
            'checkpoint_dir': checkpoint_dir,
            'dagger': parsed.dagger,

            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

            'teacher_args': {
                'task': parsed.teacher_task,
                'proxy': parsed.teacher_proxy
                },

            'student_args': {
                'resnet_model': parsed.resnet_model,
                'conditional': parsed.conditional
                },

            'data_args': {
                'dataset_dir': parsed.dataset_dir,
                'dataset_size': parsed.dataset_size,
                'batch_size': parsed.batch_size,

                'interpolate': parsed.interpolate,

                'dagger': parsed.dagger,
                'capacity': parsed.capacity,
                'episodes_per_epoch': parsed.episodes_per_epoch,

                'dagger_dataset_dir': parsed.dagger_dataset_dir
                },

            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay
                }
            }

    main(config)
