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
from .pointgoal_dataset import get_dataset
from .const import GIBSON_IDX2NAME

import wandb

ACTIONS = ['F', 'L', 'R']

def _log_visuals(rgb, goal, action, _action, loss):
    font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf', 12)
    images = list()

    goal = goal.cpu().numpy()
    r, t = goal[:,0], -np.arccos(goal[:,1])
    for i in range(min(rgb.shape[0], 64)):
        canvas = Image.fromarray(np.uint8(rgb[i].cpu()).reshape(160, 384, 3))
        draw = ImageDraw.Draw(canvas)

        draw.rectangle((0, 0, 384, 20), fill='black')
        draw.text((5, 5), f'Goal: ({r[i]:.2f}, {t[i]:.2f}), Expert: <{ACTIONS[action[i]]}>, Pred: <{ACTIONS[_action[i]]}>', font=font)
        images.append((loss[i].sum(), torch.ByteTensor(np.uint8(canvas).transpose(2, 0, 1))))

    result = torchvision.utils.make_grid([x[1] for x in \
            sorted(images, key=lambda x: x[0], reverse=True)[:32]], nrow=4)
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
    scene_correct, scene_total = np.zeros(len(GIBSON_IDX2NAME)), np.zeros(len(GIBSON_IDX2NAME))
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    correct, total = 0, 0
    tick = time.time()
    iterator = tqdm.tqdm(data, desc=desc, total=len(data), position=1, leave=None)
    for i, (scene_idx, rgb, goal, action) in enumerate(iterator):
        rgb = rgb.to(config['device'])
        goal = goal.to(config['device'])
        action = action.to(config['device'])

        _action = net(rgb, goal).logits

        loss = criterion(_action, action)
        loss_mean = loss.mean()

        correct += (action == _action.argmax(dim=1)).sum().item()
        total += rgb.shape[0]

        if is_train:
            loss_mean.backward()
            optim.step()
            optim.zero_grad()

            wandb.run.summary['step'] += 1

        losses.append(loss_mean.item())
        for s in scene_idx.long().unique():
            scene_correct[s] += (action[scene_idx==s] == _action[scene_idx==s].argmax(dim=1)).sum().item()
            scene_total[s] += (scene_idx==s).sum().item()

        metrics = {'loss': loss_mean.item(),
                   'images_per_second': rgb.shape[0] / (time.time() - tick)}
        #if i % == 0:
            #metrics['images'] = _log_visuals(rgb, goal, action, _action.argmax(dim=1), loss)
        wandb.log({('%s/%s' % (desc, k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    metrics = {f'{desc}/accuracy': correct/total}
    for idx, scene in enumerate(GIBSON_IDX2NAME):
        if scene_total[idx] > 0:
            metrics[f'{desc}/{scene}_accuracy'] = scene_correct[idx]/scene_total[idx]
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
    # NOTE: loading aux task
    #aux_net = InverseDynamics(**config['aux_model_args']).to(config['device'])
    #aux_net = TemporalDistance(**config['aux_model_args']).to(config['device'])
    aux_net = SceneLocalization(**config['aux_model_args']).to(config['device'])
    aux_net.load_state_dict(torch.load(config['aux_model'], map_location=config['device']))
    aux_net.eval() # NOTE: does this freeze the weights?

    net = PointGoalPolicyAux(aux_net, **config['model_args']).to(config['device'])
    for param in net.aux.parameters():
        param.requires_grad = False

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

        wandb.run.summary['best_accuracy'] = 0
        wandb.run.summary['best_epoch'] = 0

    for epoch in tqdm.tqdm(range(wandb.run.summary['epoch']+1, parsed.max_epoch+1), desc='epoch'):
        wandb.run.summary['epoch'] = epoch

        loss_train = train_or_eval(net, data_train, optim, True, config)
        with torch.no_grad():
            loss_val = train_or_eval(net, data_val, None, False, config)

        wandb.log({'train/loss_epoch': loss_train, 'val/loss_epoch': loss_val})
        if 'best_accuracy' in wandb.run.summary.keys() and wandb.run.summary['val/accuracy'] > wandb.run.summary['best_accuracy']:
            wandb.run.summary['best_accuracy'] = wandb.run.summary['val/accuracy']
            wandb.run.summary['best_epoch'] = epoch
            torch.save(net.state_dict(), Path(wandb.run.dir) / 'model_best.t7')

        checkpoint_project(net, optim, scheduler, config)
        if epoch % 100 == 0: # 24.5 MB for 256, 34 MB for 512
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

    # Data args.
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--dataset_size', type=float, default=1.0)
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
