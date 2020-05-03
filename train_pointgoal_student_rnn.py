import argparse
import time
from collections import defaultdict
import shutil
import gc
import time

import tqdm
import yaml
import wandb
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torchvision
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

from model import get_model
from habitat_dataset import get_dataset, HabitatDataset
from habitat_wrapper import TASKS, MODELS, MODALITIES, Rollout, get_episode, save_episode

import cProfile
from pytorch_memlab import profile


def pass_single(net, criterion, target, action, meta, config, optim=None):
    target = target.to(config['device'])
    action = action.to(config['device'])
    meta = meta.to(config['device'])

    _action = net((target, meta))
    loss = criterion(_action, action)

    if optim:
        loss.backward()
        optim.step()
        optim.zero_grad()

    return loss.mean().item()


def pass_sequence_tbptt(net, criterion, target, action, prev_action, meta, mask, config, optim=None):
    # NOTE: k1-k2 backprop, then k2 tbptt
    k1 = np.random.randint(4, 6) # frequency of TBPTT
    k2 = np.random.randint(2, 4) # length of TBPTT

    pass

""" chunk_sizes ------------------------------------------------------*
    c1: chunk size; graphs, hidden states, etc. must fit in memory    |
    c2: frequency of moving to GPU; i.e: number of batched-timesteps """
chunk_sizes = {
    'GeForce GTX 1080': { # 8 GB
        8:   (8, 120),
        16:  (6, 100), # whole sequence can fit
        32:  (4, 50),
        64:  (3, 6), # 
        128: (2, 2),  # 263.42 img/sec
    },
    'GeForce GTX 1080 Ti': { # 11 GB
        8:   (10, 120),
        16:  (8, 100),
        32:  (6, 50),
        64:  (4, 6),
        128: (2, 2)
    }
}


def pass_sequence_backprop(net, criterion, target, action, prev_action, meta, mask, config, optim=None):
    c1, c2 = chunk_sizes[config['gpu']][net.batch_size]
    #print(f'c1={c1} c2={c2}')

    sequence_loss, chunk_loss = 0, 0
    for t in range(target.shape[0]):
        #print(f't={t}')
        if optim:
            optim.zero_grad()

        if t % c2 == 0:
            if t > 0:
                _target.detach_()
                del _target
                gc.collect()
                torch.cuda.empty_cache()
            # (S x B x 256 x 256 x 3) x 4 bytes => 0.79 MB per batch-timestep
            _target = target[t:t+c2].to(config['device'], non_blocking=True)

        _action = net((_target[t%c2], meta[t], prev_action[t], mask[t]))

        loss = criterion(_action, action[t])
        chunk_loss += loss

        net.hidden_states.detach_()
        if t % c1 == 0: # how many 
            if optim:
                chunk_loss.backward()
                optim.step()

            chunk_loss.detach_() # free memory
            del chunk_loss
            gc.collect()
            torch.cuda.empty_cache()
            chunk_loss = 0

        sequence_loss += loss.item()

    # cleanup; has the accidential side-effect: last few steps have larger step
    if chunk_loss != 0:
        if optim:
            optim.zero_grad()
            chunk_loss.backward()
            optim.step()
        chunk_loss.detach_()

    # just slows things down
    target.detach_()
    del target
    gc.collect()

    return sequence_loss / mask.sum().item() # average over all real batch-sequence elements


def pass_sequence(net, criterion, target, action, prev_action, meta, mask, config, optim=None):
    net.clean() # start episode!
    net.hidden_states.detach_()

    target_batch = pad_sequence(target)
    #print('pass_sequence')
    #print(mask.sum().item(), target_batch.shape)

    action = action.to(config['device'])
    prev_action = prev_action.to(config['device'])
    meta = meta.to(config['device'])
    mask = mask.to(config['device'])

    method = config['student_args']['method']
    if method == 'backprop':
        return pass_sequence_backprop(net, criterion, target_batch, action, prev_action, meta, mask, config, optim)
    elif method == 'tbptt':
        return pass_sequence_tbptt(net, criterion, target_batch, action, prev_action, meta, mask, config, optim)


def get_target(depth, rgb, semantic, config):
    if config['student_args']['target'] == 'depth':
        return depth
    if config['student_args']['target'] == 'semantic':
        return semantic
    return rgb


def validate(net, env, data, config):
    net.eval()
    net.batch_size = config['data_args']['batch_size']
    #env.mode = 'student'

    losses = list()
    criterion = torch.nn.CrossEntropyLoss()
    tick = time.time()

    # static validation set
    for i, x in enumerate(tqdm.tqdm(data, desc='val', total=len(data), leave=False)):
        if config['student_args']['method'] == 'feedforward':
            depth, rgb, _, semantic, action, meta, _, _ = x
            loss_mean = pass_single(net, criterion, get_target(depth, rgb, semantic, config), action, meta, config, optim=None)
        else:
            depth, rgb, semantic, action, prev_action, meta, mask = x
            loss_mean = pass_sequence(net, criterion, get_target(depth, rgb, semantic, config), action, prev_action, meta, mask, config, optim=None)

        losses.append(loss_mean)
        metrics = {
            'loss': loss_mean,
            'images_per_second': mask.sum().item() / (time.time() - tick)
        }

        wandb.log(
                {('%s/%s' % ('val', k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    return np.mean(losses)


def train(net, env, data, optim, config):
    net.train()
    net.batch_size = config['data_args']['batch_size']
    #env.mode = 'teacher'

    losses = list()
    criterion = torch.nn.CrossEntropyLoss()
    tick = time.time()

    for i, x in enumerate(tqdm.tqdm(data, desc='train', total=len(data), leave=False)):
        if config['student_args']['method'] == 'feedforward':
            depth, rgb, _, semantic, action, meta, _, _ = x
            loss_mean = pass_single(net, criterion, get_target(depth, rgb, semantic, config), action, meta, config, optim=optim)
        else:
            depth, rgb, semantic, action, prev_action, meta, mask = x
            loss_mean = pass_sequence(net, criterion, get_target(depth, rgb, semantic, config), action, prev_action, meta, mask, config, optim=optim)

        losses.append(loss_mean)
        wandb.run.summary['step'] += 1
        metrics = {
            'loss': loss_mean,
            'images_per_second': mask.sum().item() / (time.time() - tick)
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
        input_channels = HabitatDataset.NUM_SEMANTIC_CLASSES
    net = get_model(**config['student_args'], input_channels=input_channels).to(config['device'])
    data_train, data_val = get_dataset(**config['data_args'], rgb=config['student_args']['target']=='rgb', semantic=config['student_args']['target']=='semantic')

    #env_train = Rollout(**config['teacher_args'], student=net, rnn=True, split='train')
    sensors = ['RGB_SENSOR']
    if config['student_args']['target'] == 'semantic': # NOTE: computing semantic is slow
        sensors.append('SEMANTIC_SENSOR')
    env_val = None
    #env_val = Rollout(task=config['teacher_args']['task'], proxy=config['student_args']['target'], mode='student', student=net, rnn=config['student_args']['rnn'], shuffle=True, split='val', dataset=config['data_args']['scene'], sensors=sensors, gpu_id=1)

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

        """
        spl_mean = all_spl[-1].mean() if len(all_spl) > 0 else 0.0
        if spl_mean > wandb.run.summary.get('best_spl', -np.inf):
            wandb.run.summary['best_spl'] = spl_mean
            wandb.run.summary['best_spl_epoch'] = wandb.run.summary['epoch']

        soft_spl_mean = all_soft_spl[-1].mean() if len(all_soft_spl) > 0 else 0.0
        if soft_spl_mean > wandb.run.summary.get('best_soft_spl', -np.inf):
            wandb.run.summary['best_soft_spl'] = soft_spl_mean
            wandb.run.summary['best_soft_spl_epoch'] = wandb.run.summary['epoch']
        """

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
    parser.add_argument('--method', type=str, choices=['feedforward', 'backprop', 'tbptt'], default='backprop', required=True)

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
            'gpu': torch.cuda.get_device_name(0),

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
                'num_workers': 1 if parsed.method != 'feedforward' else 4,

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
