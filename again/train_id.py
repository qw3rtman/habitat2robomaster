import argparse
import time
import yaml

from pathlib import Path

import tqdm
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

from .model import InverseDynamics
from .consecutive_dataset import get_dataset

import wandb

ACTIONS = ['F', 'L', 'R']

def _log_visuals(t1, t2, action, _action, loss):
    font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf', 12)
    images = list()

    rgb = torch.cat([t1, t2], dim=1)
    for i in range(min(rgb.shape[0], 64)):
        canvas = Image.fromarray(np.uint8(rgb[i].cpu()).reshape(-1, 384, 3))
        draw = ImageDraw.Draw(canvas)

        loss_i = loss[i].sum()
        draw.rectangle((0, 0, 384, 20), fill='black')
        draw.text((5, 5), 'Action: <{}> Pred: <{}>'.format(ACTIONS[action[i]], ACTIONS[_action[i]]))
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

    losses = list()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    correct, total = 0, 0
    tick = time.time()
    iterator = tqdm.tqdm(data, desc=desc, total=len(data), position=1, leave=None)
    for i, (t1, t2, action) in enumerate(iterator):
        t1 = t1.to(config['device'])
        t2 = t2.to(config['device'])
        action = action.to(config['device'])

        _action = net(t1, t2).logits

        loss = criterion(_action, action)
        loss_mean = loss.mean()

        correct += (action == _action.argmax(dim=1)).sum().item()
        total += t1.shape[0]

        if is_train:
            loss_mean.backward()
            optim.step()
            optim.zero_grad()

            wandb.run.summary['step'] += 1

        losses.append(loss_mean.item())

        metrics = {'loss': loss_mean.item(),
                   'images_per_second': t1.shape[0] / (time.time() - tick)}
        #if i % 50 == 0:
            #metrics['images'] = _log_visuals(t1, t2, action, _action.argmax(dim=1), loss)
        wandb.log({('%s/%s' % (desc, k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    wandb.log({f'{desc}/accuracy': correct/total}, step=wandb.run.summary['step'])
    return np.mean(losses)


def resume_project(net, optim, scheduler, config):
    print('Resumed at epoch %d.' % wandb.run.summary['epoch'])

    net.load_state_dict(torch.load(config['checkpoint_dir'] / 'model_latest.t7'))
    scheduler.load_state_dict(torch.load(config['checkpoint_dir'] / 'scheduler_latest.t7'))


def checkpoint_project(net, optim, scheduler, config):
    torch.save(net.state_dict(), config['checkpoint_dir'] / 'model_latest.t7')
    torch.save(scheduler.state_dict(), config['checkpoint_dir'] / 'scheduler_latest.t7')


def main(config):
    net = InverseDynamics(**config['aux_model_args']).to(config['device'])
    net.load_state_dict(torch.load(config['aux_model'], map_location=config['device']))

    #net = InverseDynamics(**config['model_args']).to(config['device'])

    data_train, data_val = get_dataset(**config['data_args'])
    optim = torch.optim.Adam(net.parameters(), **config['optimizer_args'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=0.5,
            milestones=[mult * config['max_epoch'] for mult in [0.5, 0.75]])

    project_name = 'pointgoal-id'
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
        if 'best_val_loss' in wandb.run.summary.keys() and loss_val < wandb.run.summary['best_val_loss']:
            wandb.run.summary['best_val_loss'] = loss_val
            wandb.run.summary['best_epoch'] = epoch

        checkpoint_project(net, optim, scheduler, config)
        if epoch % 1 == 0:
            torch.save(net.state_dict(), Path(wandb.run.dir) / ('model_%03d.t7' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=Path, default='checkpoints')

    # Aux model args. (pre-trained)
    parser.add_argument('--aux_model', type=Path, required=True)

    # Model args.
    parser.add_argument('--resnet_model', default='resnet18')
    parser.add_argument('--hidden_size', type=int, required=True)

    # Data args.
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=128)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()

    keys = ['resnet_model', 'hidden_size', 'lr', 'weight_decay', 'batch_size', 'description']
    run_name  = '_'.join(str(getattr(parsed, x)) for x in keys)

    checkpoint_dir = parsed.checkpoint_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
            'run_name': run_name,
            'max_epoch': parsed.max_epoch,
            'checkpoint_dir': checkpoint_dir,

            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

            'aux_model': parsed.aux_model,
            'aux_model_args': yaml.load((parsed.aux_model.parent / 'config.yaml').read_text())['model_args']['value'],

            'model_args': {
                'resnet_model': parsed.resnet_model,
                'hidden_size': parsed.hidden_size,
                'action_dim': 3
            },

            'data_args': {
                'num_workers': 8,
                'dataset_dir': parsed.dataset_dir,
                'batch_size': parsed.batch_size
                },

            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay
                }
            }

    main(config)
