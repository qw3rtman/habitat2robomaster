import argparse
import time
import yaml
from collections import defaultdict

from pathlib import Path

import tqdm
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

from .model import PointGoalPolicy, InverseDynamics, TemporalDistance, PointGoalPolicyAux, SceneLocalization
from .joint_dataset import get_dataset
from .const import GIBSON_IDX2NAME

import wandb


def train_or_eval(net, data, optim, is_train, config):
    if is_train:
        desc = 'train'
        net.train()
    else:
        desc = 'val'
        net.eval()

    losses = list()

    sl_scene_loss = defaultdict(list)
    sl_criterion = torch.nn.MSELoss(reduction='none')

    il_scene_correct = np.zeros(len(GIBSON_IDX2NAME))
    il_scene_total = np.zeros(len(GIBSON_IDX2NAME))
    il_scene_loss = defaultdict(list)
    il_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    correct, total = 0, 0
    tick = time.time()
    iterator = tqdm.tqdm(data, desc=desc, total=len(data), position=1, leave=None)
    for i, (scene_idx, rgb, localization, goal, action) in enumerate(iterator):
        rgb = rgb.to(config['device'])
        localization = localization.to(config['device'])
        goal = goal.to(config['device'])
        action = action.to(config['device'])

        sl_loss = sl_criterion(net.aux(rgb, scene_idx), localization)
        loss_mean = sl_loss.mean()

        _action = net(rgb, goal).logits
        il_loss = il_criterion(_action, action)
        loss_mean += il_loss.mean()

        correct += (action == _action.argmax(dim=1)).sum().item()
        total += rgb.shape[0]

        if is_train:
            loss_mean.backward()
            optim.step()
            optim.zero_grad()

            wandb.run.summary['step'] += 1

        losses.append(loss_mean.item())
        for s in scene_idx.long().unique():
            sl_scene_loss[s].append(sl_loss[scene_idx==s].mean().item())

            il_scene_loss[s].append(il_loss[scene_idx==s].mean().item())
            il_scene_correct[s] += (action[scene_idx==s] == _action[scene_idx==s].argmax(dim=1)).sum().item()
            il_scene_total[s] += (scene_idx==s).sum().item()

        metrics = {
            'loss': loss_mean.item(),
               'sl_loss': sl_loss.mean().item(),
               'il_loss': il_loss.mean().item(),
               'images_per_second': rgb.shape[0] / (time.time() - tick)
        }

        if i % 50 == 0:
            for idx, name in enumerate(GIBSON_IDX2NAME):
                if len(sl_scene_loss[idx]) > 0:
                    metrics[f'{desc}/{name}_sl_loss'] = np.mean(sl_scene_loss)
                    metrics[f'{desc}/{name}_il_loss'] = np.mean(il_scene_loss)

        wandb.log({('%s/%s' % (desc, k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    metrics = {f'{desc}/accuracy': correct/total}
    for idx, name in enumerate(GIBSON_IDX2NAME):
        if il_scene_total[idx] > 0:
            metrics[f'{desc}/{name}_accuracy'] = il_scene_correct[idx] / il_scene_total[idx]

    wandb.log(metrics, step=wandb.run.summary['step'])

    return np.mean(losses)


def resume_project(net, optim, scheduler, config):
    print('Resumed at epoch %d.' % wandb.run.summary['epoch'])

    net.load_state_dict(torch.load(config['checkpoint_dir'] / 'model_latest.t7'))
    scheduler.load_state_dict(torch.load(config['checkpoint_dir'] / 'scheduler_latest.t7'))


def checkpoint_project(net, optim, scheduler, config):
    torch.save(net.state_dict(), config['checkpoint_dir'] / 'model_latest.t7')
    torch.save(scheduler.state_dict(), config['checkpoint_dir'] / 'scheduler_latest.t7')


def main(config, parsed):
    aux_net = SceneLocalization(**config['model_args']).to(config['device'])
    net = PointGoalPolicyAux(aux_net, **config['model_args']).to(config['device'])

    data_train, data_val = get_dataset(**config['data_args'])
    optim = torch.optim.Adam(net.parameters(), **config['optimizer_args'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=0.5,
            milestones=[mult * config['max_epoch'] for mult in [0.5, 0.75]])

    project_name = 'pointgoal-il-aux'
    wandb.init(project=project_name, config=config, name=config['run_name'],
            resume=True, id=str(hash(config['run_name'])))
    wandb.save(str(Path(wandb.run.dir) / '*.t7'))
    if wandb.run.resumed:
        resume_project(net, optim, scheduler, config)
    else:
        wandb.run.summary['step'] = 0
        wandb.run.summary['epoch'] = 0

        wandb.run.summary['best_loss'] = 100
        wandb.run.summary['best_epoch'] = 0

    for epoch in tqdm.tqdm(range(wandb.run.summary['epoch']+1, parsed.max_epoch+1), desc='epoch'):
        wandb.run.summary['epoch'] = epoch

        loss_train = train_or_eval(net, data_train, optim, True, config)
        with torch.no_grad():
            loss_val = train_or_eval(net, data_val, None, False, config)

        wandb.log({'train/loss_epoch': loss_train, 'val/loss_epoch': loss_val})
        if loss_val < wandb.run.summary['best_loss']:
            wandb.run.summary['best_loss'] = loss_val
            wandb.run.summary['best_epoch'] = epoch
            torch.save(net.state_dict(), Path(wandb.run.dir) / 'model_best.t7')

        checkpoint_project(net, optim, scheduler, config)
        if epoch % 10 == 0: # 24.5 MB for 256, 34 MB for 512
            torch.save(net.state_dict(), Path(wandb.run.dir) / ('model_%03d.t7' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=Path, default='checkpoints')

    # Aux model args.
    parser.add_argument('--aux_model', type=Path, required=True)

    # Model args.
    parser.add_argument('--resnet_model', default='resnet50')
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--scene_bias', action='store_true', default=False)

    # Data args.
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--dataset_size', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=128)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()

    keys = ['resnet_model', 'hidden_size', 'lr', 'weight_decay', 'batch_size', 'description']
    run_name  = '_'.join(str(getattr(parsed, x)) for x in keys) + '_jointly' + ('_no-bias' if not parsed.scene_bias else '')

    checkpoint_dir = parsed.checkpoint_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
            'run_name': run_name,
            'max_epoch': parsed.max_epoch,
            'checkpoint_dir': checkpoint_dir,

            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

            'model_args': {
                'resnet_model': parsed.resnet_model,
                'hidden_size': parsed.hidden_size,
                'localization_dim': 4,
                'scene_bias': parsed.scene_bias,
                'action_dim': 3,
                'goal_dim': 3
                },

            'data_args': {
                'num_workers': 8,
                'dataset_dir': parsed.dataset_dir,
                'dataset_size': parsed.dataset_size,
                'batch_size': parsed.batch_size,
                'goal_fn': 'polar1'
                },

            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay
                }
            }

    main(config, parsed)
