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

from model import get_model
from habitat_dataset import get_dataset, HabitatDataset
from habitat_wrapper import TASKS, MODELS, MODALITIES, Rollout, get_episode, save_episode

all_success = []
all_spl = []
all_soft_spl = []
all_dfg = []
all_d_ratio = []
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 20)]

COLORS = [
    (255,   0, 232), # wall
    (116,  56, 117)  # floor
]

# NUM_EPISODES, VIDEO_FREQ, EPOCH_FREQ
rollout_freq = {
    'castle': [50, 10, 5],
    'office': [100, 20, 5],
    'mp3d': [50, 10, 25],
    'gibson': [50, 10, 25]
}

def _get_box(all_x, EPOCH_FREQ):
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

def pass_single(net, criterion, rgb, seg, action, meta, config, optim=None):
    if config['student_args']['target'] == 'semantic':
        target = seg.to(config['device'])
    else:
        target = rgb.to(config['device'])
    action = action.to(config['device'])
    meta = meta.to(config['device'])

    _action = net((target, meta))
    loss = criterion(_action, action)

    if optim:
        loss.backward()
        optim.step()
        optim.zero_grad()

    return loss.mean().item()

def pass_sequence(net, criterion, rgb, seg, action, prev_action, meta, mask, config, optim=None):
    net.clean() # start episode!

    method = config['student_args']['method']
    if method == 'tbptt':
        #k1 = 20
        #k2 = 10
        # NOTE: k1-k2 backprop, then k2 tbptt
        k1 = np.random.randint(7, 12) # frequency of TBPTT
        k2 = np.random.randint(2, 6) # length of TBPTT

    if config['student_args']['target'] == 'semantic':
        target = seg.to(config['device'])
    else:
        target = rgb.to(config['device'])
    action = action.to(config['device'])
    prev_action = prev_action.to(config['device'])
    meta = meta.to(config['device'])
    mask = mask.to(config['device'])

    max_sequence_length = target.shape[0]
    # NOTE: prevent out of memory; batch_size=8 can do 60 on 1080 Ti,
    #                              batch_size=4 can do 83 on 1080, scales linearly
    total_memory = torch.cuda.get_device_properties(config['device']).total_memory
    if total_memory > 9e9: # 1080 Ti (11718230016)
        sequence_length_capacity = int((240//config['data_args']['batch_size']) - 10)
    else: #                  1080    (8513978368)
        sequence_length_capacity = int((200//config['data_args']['batch_size']) - 10)
    #print(f'sequence length capacity: {sequence_length_capacity}')

    tbptt = method in ['tbptt', 'wwtbptt']
    if method == 'tbptt':
        truncate_indices = np.arange(0, target.shape[0])
        indices = np.where((truncate_indices % k1 == k2)|(truncate_indices % k1 == 0))[0]
    elif method == 'wwtbptt': # https://arxiv.org/abs/1702.07600
        window_start = np.random.randint(max_sequence_length * 0.8)
        window_end = np.random.randint(window_start, min(window_start+20, max_sequence_length))
        indices = [window_start, window_end]
        with torch.no_grad(): # move to start of window
            for t in range(window_start):
                net((target[t], meta[t], prev_action[t], mask[t]))
    else:
        indices = range(0, max_sequence_length, sequence_length_capacity) # chunking

    sequence_loss = 0
    for i, start in enumerate(indices):
        if method in ['tbptt', 'wwtbptt']:
            end = indices[i+1] if i+1 < len(indices) else min(start+20, max_sequence_length)
            if 0 in action[start:end]: # we really want to capture the closing move
                tbptt = True
        else:
            end = min(start+sequence_length_capacity, max_sequence_length)

        chunk_loss = 0
        net.hidden_states.detach_()
        for t in range(start, end):
            #alloc = torch.cuda.memory_allocated(0)
            #print(f's={t}, alloc={alloc}, free={total_memory-alloc}, tbptt={tbptt}')
            _action = net((target[t], meta[t], prev_action[t], mask[t]))

            loss = criterion(_action, action[t])
            chunk_loss += loss
            if not tbptt:
                net.hidden_states.detach_()

        if optim and hasattr(chunk_loss, 'backward') and chunk_loss != 0:
            chunk_loss.backward()
            optim.step()
            optim.zero_grad()

        if hasattr(chunk_loss, 'backward') and chunk_loss != 0:
            chunk_loss.detach_() # free memory
            sequence_loss += chunk_loss.item()

        if method not in ['tbptt', 'wwtbptt']:
            tbptt = not tbptt

    return sequence_loss / max_sequence_length # loss mean


def validate(net, env, data, config):
    NUM_EPISODES, VIDEO_FREQ, EPOCH_FREQ = rollout_freq[config['data_args']['scene']]

    net.eval()
    net.batch_size = config['data_args']['batch_size']
    env.mode = 'student'

    losses = list()
    criterion = torch.nn.CrossEntropyLoss()
    tick = time.time()

    # static validation set
    for i, x in enumerate(tqdm.tqdm(data, desc='val', total=len(data), leave=False)):
        if config['student_args']['method'] == 'feedforward':
            rgb, _, seg, action, meta, _, _ = x
            loss_mean = pass_single(net, criterion, rgb, seg, action, meta, config, optim=None)
        else:
            rgb, seg, action, prev_action, meta, mask = x
            loss_mean = pass_sequence(net, criterion, rgb, seg, action, prev_action, meta, mask, config, optim=None)
        losses.append(loss_mean)

        if config['student_args']['target'] == 'semantic':
            num_images = np.prod(seg.shape[:-3])
        else:
            num_images = np.prod(rgb.shape[:-3])

        metrics = {
            'loss': loss_mean,
            'images_per_second': num_images / (time.time() - tick)
        }

        wandb.log(
                {('%s/%s' % ('val', k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    # rollout
    # NOTE: slide back iterator to 1st episode so we always validate over same episodes
    # castle: 50 val episodes
    # office: 495 ...
    env.env.episode_iterator._iterator = iter(env.env.episode_iterator.episodes)
    net.batch_size = 1
    if wandb.run.summary['epoch'] % EPOCH_FREQ == 0:
        distance_to_goal = np.zeros(NUM_EPISODES)
        distance_from_goal = np.zeros(NUM_EPISODES)
        d_ratio = np.zeros(NUM_EPISODES)
        success = np.zeros(NUM_EPISODES)
        spl = np.zeros(NUM_EPISODES)
        soft_spl = np.zeros(NUM_EPISODES)
        avg_value = np.zeros(NUM_EPISODES)

        for ep in range(NUM_EPISODES):
            value = []
            images = []

            if config['student_args']['method'] != 'feedforward':
                net.clean()

            for i, step in enumerate(get_episode(env)):
                if i == 0:
                    distance_to_goal[ep] = env.env.get_metrics()['distance_to_goal'] #np.linalg.norm(start[[0,2]]-goal[[0,2]])
                if config['student_args']['method'] != 'feedforward':
                    value.append(net.value.item())
                if ep % VIDEO_FREQ == 0:
                    frame = Image.fromarray(step['rgb'])
                    draw = ImageDraw.Draw(frame)

                    if config['student_args']['target'] == 'semantic': # overlay road
                        classes = HabitatDataset._make_semantic(step['semantic'])

                        for _class in range(classes.shape[-1]):
                            label = Image.new('RGB', frame.size, COLORS[_class])
                            mask = Image.fromarray(255 * np.uint8(classes[:,:,_class]))
                            frame = Image.composite(label,frame,mask).convert('RGB')

                    font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf', 18)
                    direction = env.get_direction()
                    draw.rectangle((0, 0, 255, 20), fill='black')
                    draw.text((0, 0), '({: <5.1f}, {: <5.1f}) {: <4.1f}'.format(*direction, np.linalg.norm(direction)), fill='white', font=font)

                    images.append(np.transpose(np.uint8(frame), (2, 0, 1)))

            env_metrics = env.env.get_metrics()
            start = np.array(env.env.current_episode.start_position)
            goal = np.array(env.env.current_episode.goals[0].position)
            curr = np.array(env.state.sensor_states['rgb'].position) # NOTE: these sensor states somehow match with start/end

            distance_from_goal[ep] = env_metrics['distance_to_goal'] #np.linalg.norm(curr[[0,2]]-goal[[0,2]])
            d_ratio[ep] = distance_to_goal[ep] / (distance_from_goal[ep] + 0.00001)
            success[ep] = env_metrics['success']
            spl[ep] = env_metrics['spl']
            soft_spl[ep] = env_metrics['softspl']
            avg_value[ep] = np.mean(value)

            metrics = {}
            if ep == NUM_EPISODES - 1:
                all_success.append(success)
                success_mean = np.mean(success)
                metrics['success_mean'] = success_mean
                metrics['success_std'] = np.std(success)
                metrics['success_median'] = np.median(success)
                metrics['success'] = wandb.Histogram(success)
                metrics['success_box'] = _get_box(all_success, EPOCH_FREQ)
                metrics['within_0.5'] = (distance_from_goal < 0.5).mean()
                metrics['within_1.0'] = (distance_from_goal < 1.0).mean()

                all_spl.append(spl)
                spl_mean = np.mean(spl)
                metrics['spl_mean'] = spl_mean
                metrics['spl_std'] = np.std(spl)
                metrics['spl_median'] = np.median(spl)
                metrics['spl'] = wandb.Histogram(spl)
                metrics['spl_box'] = _get_box(all_spl, EPOCH_FREQ)

                all_soft_spl.append(soft_spl)
                soft_spl_mean = np.mean(soft_spl)
                metrics['soft_spl_mean'] = soft_spl_mean
                metrics['soft_spl_std'] = np.std(soft_spl)
                metrics['soft_spl_median'] = np.median(soft_spl)
                metrics['soft_spl'] = wandb.Histogram(soft_spl)
                metrics['soft_spl_box'] = _get_box(all_soft_spl, EPOCH_FREQ)

                if config['student_args']['method'] != 'feedforward':
                    metrics['value_mean'] = np.mean(avg_value)

                dtg_mean = np.mean(distance_to_goal)
                metrics['dtg_mean'] = dtg_mean
                metrics['dtg_median'] = np.median(distance_to_goal)
                metrics['dtg'] = wandb.Histogram(distance_to_goal)

                all_dfg.append(distance_from_goal)
                dfg_mean = np.mean(distance_from_goal)
                metrics['dfg_mean'] = dfg_mean
                metrics['dfg_median'] = np.median(distance_from_goal)
                metrics['dfg_box'] = _get_box(all_dfg, EPOCH_FREQ)
                metrics['dfg'] = wandb.Histogram(distance_from_goal)

                # how close are we to goal relative to the starting distance?
                all_d_ratio.append(d_ratio)
                d_ratio_mean = np.mean(d_ratio)
                metrics['d_ratio_mean'] = d_ratio_mean
                metrics['d_ratio_box'] = _get_box(all_d_ratio, EPOCH_FREQ)

                # difficulty of episodes has big impact on SPL/success, so normalize
                metrics['spl_dtg_mean'] = dtg_mean * spl_mean
                metrics['success_dtg_mean'] = dtg_mean * success_mean

                metrics['distance_to_goal_vs_success'] = _get_hist2d(distance_to_goal, success)
                metrics['distance_to_goal_vs_spl'] = _get_hist2d(distance_to_goal, spl)
                metrics['distance_to_goal_vs_soft_spl'] = _get_hist2d(distance_to_goal, soft_spl)
                metrics['distance_from_goal_vs_success'] = _get_hist2d(distance_from_goal, success)

            if ep % VIDEO_FREQ == 0 and len(images) > 0:
                fig = go.Figure(data=[go.Scatter(
                    x=np.arange(1, len(value)+1),
                    y=value
                )])
                metrics[f'values_{(ep//VIDEO_FREQ)+1}'] = fig
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


    for i, x in enumerate(tqdm.tqdm(data, desc='train', total=len(data), leave=False)):
        # rgb.shape
        # sequence, batch, ...

        if config['student_args']['method'] == 'feedforward':
            rgb, _, seg, action, meta, _, _ = x
            loss_mean = pass_single(net, criterion, rgb, seg, action, meta, config, optim=optim)
        else:
            rgb, seg, action, prev_action, meta, mask = x
            loss_mean = pass_sequence(net, criterion, rgb, seg, action, prev_action, meta, mask, config, optim=optim)
        losses.append(loss_mean)

        wandb.run.summary['step'] += 1

        if config['student_args']['target'] == 'semantic':
            num_images = np.prod(seg.shape[:-3])
        else:
            num_images = np.prod(rgb.shape[:-3])

        metrics = {
            'loss': loss_mean,
            'images_per_second': num_images / (time.time() - tick)
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
    input_channels = 3
    if config['student_args']['target'] == 'semantic':
        input_channels = 2
    net = get_model(**config['student_args'], input_channels=input_channels).to(config['device'])
    data_train, data_val = get_dataset(**config['data_args'], rgb=config['student_args']['target']=='rgb', semantic=config['student_args']['target']=='semantic')

    #env_train = Rollout(**config['teacher_args'], student=net, rnn=True, split='train')
    sensors = ['RGB_SENSOR']
    if config['student_args']['target'] == 'semantic': # NOTE: computing semantic is slow
        sensors.append('SEMANTIC_SENSOR')
    env_val = Rollout(task=config['teacher_args']['task'], proxy=config['student_args']['target'], mode='student', student=net, rnn=config['student_args']['rnn'], shuffle=True, split='val', dataset=config['data_args']['scene'], sensors=sensors)

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

        spl_mean = all_spl[-1].mean() if len(all_spl) > 0 else 0.0
        if spl_mean > wandb.run.summary.get('best_spl', -np.inf):
            wandb.run.summary['best_spl'] = spl_mean
            wandb.run.summary['best_spl_epoch'] = wandb.run.summary['epoch']

        soft_spl_mean = all_soft_spl[-1].mean() if len(all_soft_spl) > 0 else 0.0
        if soft_spl_mean > wandb.run.summary.get('best_soft_spl', -np.inf):
            wandb.run.summary['best_soft_spl'] = soft_spl_mean
            wandb.run.summary['best_soft_spl_epoch'] = wandb.run.summary['epoch']

        if epoch % 10 == 0:
            torch.save(net.state_dict(), Path(wandb.run.dir) / ('model_%03d.t7' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=Path, default='checkpoints')

    # Teacher args.
    parser.add_argument('--teacher_task', choices=TASKS, required=True)
    parser.add_argument('--proxy', choices=MODELS.keys(), required=True)

    # Student args.
    parser.add_argument('--target', choices=MODALITIES, required=True)
    parser.add_argument('--resnet_model', choices=['resnet18', 'resnet50', 'resneXt50', 'se_resnet50', 'se_resneXt101', 'se_resneXt50'])
    parser.add_argument('--method', type=str, choices=['feedforward', 'backprop', 'tbptt', 'wwtbptt'], default='backprop', required=True)

    # Data args.
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--dataset_size', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--reduced', action='store_true')
    parser.add_argument('--augmentation', action='store_true')

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()

    run_name = '-'.join(map(str, [
        parsed.resnet_model,
        'bc', parsed.method,                                                                                  # training paradigm
        f'{parsed.proxy}2{parsed.target}',                                                                    # modalities
        parsed.scene, 'aug' if parsed.augmentation else 'noaug', 'reduced' if parsed.reduced else 'original', # dataset
        parsed.dataset_size, parsed.batch_size, parsed.lr, parsed.weight_decay                                # boring stuff
    ])) + f'-v{parsed.description}'

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
                'proxy': parsed.proxy
                },

            'student_args': {
                'target': parsed.target,
                'resnet_model': parsed.resnet_model,
                'method': parsed.method,
                'rnn': parsed.method != 'feedforward',
                'conditional': True,
                'batch_size': parsed.batch_size
                },

            'data_args': {
                'scene': parsed.scene,                         # the simulator's evaluation scene
                'dataset_dir': parsed.dataset_dir,
                'dataset_size': parsed.dataset_size,
                'batch_size': parsed.batch_size,
                'rnn': parsed.method != 'feedforward',

                'reduced': parsed.reduced,
                'augmentation': parsed.augmentation
                },

            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay
                }
            }

    main(config)
