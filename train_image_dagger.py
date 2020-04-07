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

from PIL import Image, ImageDraw

from model import get_model
from habitat_dataset import get_dataset, HabitatDataset, ACTIONS
from habitat_wrapper import TASKS, MODELS, Rollout, get_episode


def validate(net, env, config):
    NUM_EPISODES = 15
    VIDEO_FREQ   = 3

    env.eval()
    net.eval()

    losses = list()
    criterion = torch.nn.BCEWithLogitsLoss()
    tick = time.time()

    total_lwns = 0
    for ep in range(NUM_EPISODES):
        images = []
        loss, lwns, j = 0, 0, 0
        for i, step in enumerate(env.rollout()):
            _action = step['pred_action_logits']
            action = ACTIONS[step['true_action']].clone().unsqueeze(dim=0)

            _action.to(config['device'])
            action.to(config['device'])

            loss_mean = criterion(_action, action).mean()
            loss += loss_mean.item()
            losses.append(loss_mean.item())

            if step['is_stuck']:
                lwns = max(lwns, j)
                j = 0
            j += 1

            if ep % VIDEO_FREQ == 0:
                images.append(np.transpose(step['rgb'], (2, 0, 1)))

        total_lwns += lwns
        if ep == NUM_EPISODES - 1 and config['teacher']['teacher_args'] == 'dontcrash':
            metrics['LWNS'] = total_lwns / NUM_EPISODES

        metrics = {'loss': loss/(i+1), 'images_per_second': (i+1)/(time.time()-tick)}
        if ep % VIDEO_FREQ == 0:
            metrics[f'video_{ep}'] = wandb.Video(np.array(images), fps=30, format='mp4')

        wandb.log(
                {('%s/%s' % ('val', k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    return np.mean(losses)


def train(net, env, data, optim, config):
    if config['dagger']:
        env.eval()

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
            episode_dir = config['data_args']['dagger_dataset_dir'] / 'train' / '{:06}'.format(int(summary['ep']))
            if episode_dir.exists():
                shutil.rmtree(episode_dir, ignore_errors=True)

            episode_dir.mkdir(parents=True, exist_ok=True)
            get_episode(env, episode_dir)
            data.add_episode(HabitatDataset(episode_dir))

            num_episodes += 1

        summary['ep'] += num_episodes
        pd.DataFrame([summary]).to_csv(summary_file, index=False)

        for i, episode in enumerate(data.episodes.datasets):
            episode.episode_idx = i

        # cleanup after dagger; make student trainable
        data.post_dagger()

        episode_loss = np.zeros(len(data.episodes.datasets))
        episode_step = np.ones(len(data.episodes.datasets)) # prevent divide by zero

    losses = list()
    criterion = torch.nn.BCEWithLogitsLoss()
    tick = time.time()

    net.train()
    for i, (rgb, _, _, action, _, episode_idx) in enumerate(tqdm.tqdm(data, desc='train', total=len(data), leave=False)):
        rgb = rgb.to(config['device'])
        action = action.to(config['device'])

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

        metrics = dict()
        metrics['loss'] = loss_mean.item()
        metrics['images_per_second'] = rgb.shape[0] / (time.time() - tick)

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
    data_train, _ = get_dataset(**config['data_args'])

    (config['data_args']['dagger_dataset_dir'] / 'train').mkdir(parents=True, exist_ok=True)
    ACTIONS.to(config['device'])

    env = Rollout(**config['teacher_args'], model=net)

    optim = torch.optim.Adam(net.parameters(), **config['optimizer_args'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim,
            milestones=[config['max_epoch'] * 0.5, config['max_epoch'] * 0.75],
            gamma=0.5)

    wandb.init(project='habitat-{}-{}-student'.format(
        config['teacher_args']['task'], config['teacher_args']['proxy']
        ), config=config, id=config['run_name'], resume='auto')
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
            loss_val = validate(net, env, config)

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
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--capacity', type=int, default=1000)         # if DAgger
    parser.add_argument('--episodes_per_epoch', type=int, default=50) # if DAgger
    parser.add_argument('--dagger_dataset_dir', type=Path)            # if DAgger

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()

    keys = [
        'resnet_model', 'conditional', 'lr', 'weight_decay', # student: model, training
        'dataset_size', 'batch_size', 'capacity',            # dataset: training student
        'episodes_per_epoch',                                # dataset: generating via teacher
        'teacher_task', 'teacher_proxy', 'dagger']           # teacher: proxy task, dagger
    run_name = '_'.join(str(getattr(parsed, x)) for x in keys) + '_v6.1'

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
                'proxy': parsed.teacher_proxy,
                'dagger': parsed.dagger
                },

            'student_args': {
                'resnet_model': parsed.resnet_model,
                'conditional': parsed.conditional
                },

            'data_args': {
                'dataset_dir': parsed.dataset_dir,
                'dataset_size': parsed.dataset_size,
                'batch_size': parsed.batch_size,

                'dagger': parsed.dagger,
                'capacity': parsed.capacity,
                'per_epoch': parsed.per_epoch,

                'dagger_dataset_dir': parsed.dagger_dataset_dir
                },

            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay
                }
            }

    main(config)
