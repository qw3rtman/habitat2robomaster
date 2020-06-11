import argparse
import time

from pathlib import Path

import tqdm
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw

from .source_dataset import get_dataset
from .goal_prediction import GoalPredictionModel
from .util import C, make_onehot

import wandb

ACTIONS = ['S', 'F', 'L', 'R']
BACKGROUND = (0,47,0)
COLORS = [
    #(0,47,0),
    (102,102,102),
    (253,253,17)
]

def _log_visuals(segmentation, loss, waypoints, _waypoints, action):
    images = list()
    for i in range(min(segmentation.shape[0], 64)):
        canvas = np.zeros((segmentation.shape[-2], segmentation.shape[-1], 3), dtype=np.uint8)
        canvas[...] = BACKGROUND

        for c in range(min(segmentation.shape[1], len(COLORS))):
            canvas[segmentation.cpu()[i, c, :, :] > 0] = COLORS[c]

        canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas)

        for x, y in waypoints[i].detach().cpu().numpy().copy():
            draw.ellipse((x-2, y-2, x+2, y+2), fill=(0, 0, 255))

        for x, y in _waypoints[i].detach().cpu().numpy().copy():
            draw.ellipse((x-2, y-2, x+2, y+2), fill=(255, 0, 0)) 

        loss_i = loss[i].sum()
        draw.text((5, 10), 'Loss: %.2f' % loss_i)
        draw.text((5, 20), ' (%s) ' % ACTIONS[action[i]])
        images.append((loss_i, torch.ByteTensor(np.uint8(canvas).transpose(2, 0, 1))))

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images[:32]], nrow=4)
    result = [wandb.Image(result.numpy().transpose(1, 2, 0))]

    return result


def train_or_eval(net, data, optim, is_train, config):
    if is_train:
        desc = 'train'
        net.train()
    else:
        desc = 'val'
        net.eval()

    tick = time.time()
    losses = list()

    iterator = tqdm.tqdm(data, desc=desc, total=len(data), position=1, leave=None)
    for i, (target, actions, goals, waypoints) in enumerate(iterator):
        target = target.to(config['device'])
        target = target.reshape(config['data_args']['batch_size'], C, 160, 384)

        waypoints = waypoints.to(config['device'])
        waypoints = (waypoints + 1) * config['data_args']['zoom'] / 2

        _waypoints = net(target, actions) # [-1, 1]
        _waypoints[..., 0] = (_waypoints[..., 0] + 1) * 384 / 2
        _waypoints[..., 1] = (_waypoints[..., 1] + 1) * 160 / 2

        loss = torch.abs(waypoints - _waypoints)
        loss_mean = loss.sum((1, 2)).mean()
        losses.append(loss_mean.item())

        if is_train:
            loss_mean.backward()
            optim.step()
            optim.zero_grad()

            wandb.run.summary['step'] += 1

        metrics = {'loss': loss_mean.item(),
                   'images_per_second': target.shape[0] / (time.time() - tick)}
        if i % 100 == 0:
            metrics['images'] = _log_visuals(target, loss, waypoints, _waypoints, actions)
        wandb.log({('%s/%s' % (desc, k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    return np.mean(losses)


def resume_project(net, optim, scheduler, config):
    print('Resumed at epoch %d.' % wandb.run.summary['epoch'])

    net.load_state_dict(torch.load(config['checkpoint_dir'] / 'model_latest.t7'))
    scheduler.load_state_dict(torch.load(config['checkpoint_dir'] / 'scheduler_latest.t7'))


def checkpoint_project(net, optim, scheduler, config):
    torch.save(net.state_dict(), config['checkpoint_dir'] / 'model_latest.t7')
    torch.save(scheduler.state_dict(), config['checkpoint_dir'] / 'scheduler_latest.t7')


def main(config):
    net = GoalPredictionModel(**config['model_args']).to(config['device'])
    data_train, data_val = get_dataset(**config['data_args'])
    optim = torch.optim.Adam(net.parameters(), **config['optimizer_args'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=0.5,
            milestones=[mult * config['max_epoch'] for mult in [0.5, 0.75]])

    project_name = 'pointgoal-{}-goal-prediction'.format(config['data_args']['target_type'])
    wandb.init(project=project_name, config=config, name=config['run_name'],
            resume=True, id=str(hash(config['run_name'])))
    wandb.save(str(Path(wandb.run.dir) / '*.t7'))
    if wandb.run.resumed:
        resume_project(net, optim, scheduler, config)
    else:
        wandb.run.summary['step'] = 0
        wandb.run.summary['epoch'] = 0
        wandb.run.summary['best_epoch'] = 0

    for epoch in tqdm.tqdm(range(wandb.run.summary['epoch']+1, config['max_epoch']+1), desc='epoch'):
        wandb.run.summary['epoch'] = epoch

        loss_train = train_or_eval(net, data_train, optim, True, config)
        with torch.no_grad():
            loss_val = train_or_eval(net, data_val, None, False, config)

        wandb.log({'train/loss_epoch': loss_train, 'val/loss_epoch': loss_val})
        if loss_val < wandb.run.summary.get('best_val_loss', np.inf):
            wandb.run.summary['best_val_loss'] = loss_val
            wandb.run.summary['best_epoch'] = epoch

        checkpoint_project(net, optim, scheduler, config)
        if epoch % 10 == 0:
            torch.save(net.state_dict(), Path(wandb.run.dir) / ('model_%03d.t7' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=Path, default='checkpoints')

    # Model args.
    parser.add_argument('--resnet_model', default='resnet18')
    parser.add_argument('--temperature', type=float, default=1.0)

    # Data args.
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--target', required=True, choices=['semantic', 'depth'])
    parser.add_argument('--scene', required=True)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()

    keys = ['resnet_model', 'lr', 'weight_decay', 'batch_size', 'temperature', 'target', 'description']
    run_name  = '_'.join(str(getattr(parsed, x)) for x in keys)

    checkpoint_dir = parsed.checkpoint_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
            'run_name': run_name,
            'max_epoch': parsed.max_epoch,
            'checkpoint_dir': checkpoint_dir,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'model_args': {
                'input_channel': C,
                'resnet_model': parsed.resnet_model,
                'temperature': parsed.temperature,
                'steps': 8
                },
            'data_args': {
                'num_workers': 8,
                'target_type': parsed.target,
                'dataset_dir': parsed.dataset_dir,
                'batch_size': parsed.batch_size,
                'scene': parsed.scene,
                'zoom': 3,
                'steps': 8
                },
            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay
                }
            }

    main(config)
