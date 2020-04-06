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

from model import *
from eval import _get_network, NETWORKS
from habitat_dataset import get_dataset, HabitatDataset
from habitat_wrapper import models, Rollout, get_episode


def validate(net, env, config):
    net.eval()
    env.model = net
    env.evaluate = True

    losses = list()
    criterion = torch.nn.BCEWithLogitsLoss() # if 'ddppo' in config['network'] else torch.nn.L1Loss(reduction='none')
    tick = time.time()

    ACTIONS = torch.eye(4, device=config['device'])

    # NOTE: make actions based on our policy, evaluate/regress against PPOAgent policy
    for ep in range(5): # episodes; 300/2000 = 0.15
        loss = 0
        images = []

        longest_no_stuck = 0
        j = 0
        for i, step in enumerate(env.rollout()):
            _action = step['pred_action_logits']
            action = ACTIONS[step['true_action']].unsqueeze(dim=0)

            _action.to(config['device'])
            action.to(config['device'])

            loss_mean = criterion(_action, action).mean()
            loss += loss_mean.item()
            losses.append(loss_mean.item())

            if step['is_stuck']:
                longest_no_stuck = max(longest_no_stuck, j)
                j = 0
            j += 1

            if ep % 1 == 0:
                images.append(np.transpose(step['rgb'], (2, 0, 1)))

        metrics = {
            'loss': loss / (i+1),
            'images_per_second': i / (time.time() - tick),
            'longest_with_no_stuck': longest_no_stuck
        }

        if ep % 1 == 0:
            metrics[f'video_{ep}'] = wandb.Video(np.array(images), fps=30, format='mp4')

        wandb.log(
                {('%s/%s' % ('val', k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    return np.mean(losses)


def train(net, env, data, optim, config):
    net.train()
    env.evaluate = False

    losses = list()
    criterion = torch.nn.BCEWithLogitsLoss() # if 'ddppo' in config['network'] else torch.nn.L1Loss(reduction='none')
    tick = time.time()

    summary = defaultdict(float)
    summary['ep'] = 1
    if (config['data_args']['dataset_dir'] / 'summary.csv').exists():
        summary = pd.read_csv(config['data_args']['dataset_dir'] / 'summary.csv').iloc[0]

    # rollout some datasets; aggregate
    num_samples, num_episodes = 0, 0
    while not (num_samples > 1000 and num_episodes > 50): # until both of these conditions are met...
        episode_dir = config['data_args']['dataset_dir'] / 'train' / '{:06}'.format(int(summary['ep']))
        if episode_dir.exists():
            shutil.rmtree(episode_dir, ignore_errors=True)
        episode_dir.mkdir(parents=True, exist_ok=True)

        get_episode(env, episode_dir, evaluate=False, incomplete_ok=True)
        episode = HabitatDataset(episode_dir, apply_transform=config['data_args']['apply_transform'])
        data.add_episode(episode)
        num_samples += len(episode)
        num_episodes += 1
        summary['ep'] += 1

        pd.DataFrame([summary]).to_csv(config['data_args']['dataset_dir'] / 'summary.csv', index=False)

    for i, episode in enumerate(data.episodes.datasets):
        episode.episode_idx = i

    # cleanup after dagger
    data.post_dagger()

    episode_loss = np.zeros(len(data.episodes.datasets))
    episode_step = np.ones(len(data.episodes.datasets)) # prevent divide by zero
    for i, (rgb, _, _, action, _, episode_idx) in enumerate(tqdm.tqdm(data, desc='train', total=len(data), leave=False)):
        rgb = rgb.to(config['device'])
        action = action.to(config['device'])

        _action = net((rgb,) if 'direct' in config['network'] else (rgb, meta))

        loss = criterion(_action, action)
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
    net = _get_network(config['network']).to(config['device'])
    data_train = get_dataset(**config['data_args'])

    optim = torch.optim.Adam(net.parameters(), **config['optimizer_args'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim,
            milestones=[config['max_epoch'] * 0.5, config['max_epoch'] * 0.75],
            gamma=0.5)

    wandb.init(project='habitat-ppo-depth-student-dagger', config=config, id=config['run_name'], resume='auto')
    wandb.save(str(Path(wandb.run.dir) / '*.t7'))

    if wandb.run.resumed:
        resume_project(net, optim, scheduler, config)
    else:
        wandb.run.summary['step'] = 0
        wandb.run.summary['epoch'] = 0

    env = Rollout(config['teacher_args']['input_type'], dagger=True, model=net)
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
    parser.add_argument('--network', choices=NETWORKS, required=True)

    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=Path, default='checkpoints')

    # Model args.
    parser.add_argument('--resnet_model', default='SE-ResNeXt-50')
    parser.add_argument('--pretrained', default=False, action='store_true')

    # Data args.
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=128)

    # Teacher args.
    parser.add_argument('--input_type', choices=models.keys(), required=True)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()

    keys = ['resnet_model', 'lr', 'weight_decay', 'batch_size']
    run_name = '_'.join(str(getattr(parsed, x)) for x in keys) + '_v5.8'

    checkpoint_dir = parsed.checkpoint_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
            'run_name': run_name,
            'max_epoch': parsed.max_epoch,
            'checkpoint_dir': checkpoint_dir,
            'network': parsed.network,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

            'model_args': {
                'resnet_model': parsed.resnet_model,
                'pretrained': parsed.pretrained,
                'input_channel': 3
                },

            'data_args': {
                'dataset_dir': parsed.dataset_dir,
                'batch_size': parsed.batch_size,
                'apply_transform': 'ddppo' not in parsed.network,
                'dagger': True,
                'capacity': 2000
                },

            'teacher_args': {
                'input_type': parsed.input_type
                },

            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay
                }
            }

    main(config)
