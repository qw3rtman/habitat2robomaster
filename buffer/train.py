import argparse
from pathlib import Path
import time

import tqdm
import wandb

import torch
import torchvision
import numpy as np

import sys
sys.path.append('/u/nimit/Documents/robomaster/habitat2robomaster')

from model import ConditionalImitation
from habitat_wrapper import MODELS, MODALITIES, Rollout, replay_episode
from frame_buffer import ReplayBuffer, LossSampler


def loop(net, data, replay_buffer, env, optim, config, mode='train'):
    if mode == 'train':
        net.train()
    else:
        net.eval()

    losses = list()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    tick = time.time()
    for idx, target, goal, prev_action, action in tqdm.tqdm(data, desc=mode, total=len(data), leave=False):
        target = torch.as_tensor(target, device=config['device'], dtype=torch.float32)
        goal = torch.as_tensor(goal, device=config['device'], dtype=torch.float32)
        prev_action = torch.as_tensor(prev_action, device=config['device'], dtype=torch.int64)
        action = torch.as_tensor(action, device=config['device'], dtype=torch.int64)

        _action = net((target, goal, prev_action)).logits
        loss = criterion(_action, action)
        loss_mean = loss.mean()

        if mode == 'train':
            loss_mean.backward()
            optim.step()
            optim.zero_grad()

        replay_buffer.update_loss(idx, loss.detach().cpu().numpy())
        losses.append(loss_mean.item())

        wandb.run.summary['step'] += 1
        metrics = {'loss': loss_mean.item(),
                   'images_per_second': target.shape[0] / (time.time() - tick)}
        wandb.log({('%s/%s' % ('train', k)): v for k, v in metrics.items()},
                   step=wandb.run.summary['step'])
        tick = time.time()

    return np.mean(losses)


def resume_project(net, scheduler, config):
    print('Resumed at epoch %d.' % wandb.run.summary['epoch'])

    net.load_state_dict(torch.load(config['checkpoint_dir'] / 'model_latest.t7'))
    scheduler.load_state_dict(torch.load(config['checkpoint_dir'] / 'scheduler_latest.t7'))


def checkpoint_project(net, scheduler, config):
    torch.save(net.state_dict(), config['checkpoint_dir'] / 'model_latest.t7')
    torch.save(scheduler.state_dict(), config['checkpoint_dir'] / 'scheduler_latest.t7')


def main(config):
    net = ConditionalImitation(**config['student_args'], goal_size=3).to(config['device'])
    optim = torch.optim.Adam(net.parameters(), **config['optimizer_args'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=0.5,
            milestones=[config['max_epoch'] * 0.5, config['max_epoch'] * 0.75])

    # TODO: should support gibson+mp3d, not just gibson
    env = Rollout('pointgoal', config['teacher_args']['proxy'],
            config['student_args']['target'], mode='teacher',
            shuffle=False, split='train', dataset='gibson')
    replay_buffer = ReplayBuffer(2**18, dshape=(256,256,3), dtype=torch.uint8)
    sampler = LossSampler(replay_buffer, config['data_args']['batch_size'])
    dataset = replay_buffer.get_dataset()
    data = torch.utils.data.DataLoader(dataset, num_workers=0, pin_memory=True, batch_sampler=sampler)

    project_name = 'habitat-pointgoal-{}-student'.format(config['teacher_args']['proxy'])
    wandb.init(project=project_name, config=config, id=config['run_name'], resume='auto')
    wandb.save(str(Path(wandb.run.dir) / '*.t7'))
    if wandb.run.resumed:
        resume_project(net, scheduler, config)
    else:
        wandb.run.summary['step'] = 0
        wandb.run.summary['epoch'] = 0

    for epoch in tqdm.tqdm(range(wandb.run.summary['epoch']+1, config['max_epoch']+1), desc='epoch'):
        wandb.run.summary['epoch'] = epoch

        if epoch > 1:
            env.env._episode_iterator.max_scene_repeat_episodes = 16

        # 4 episodes per scenes to populate the buffer; then 32 scenes per epoch
        for _ in range(64):#512 if epoch > 1 else 1328):
            replay_episode(env, replay_buffer, score_by=net)

        loss_train = loop(net, data, replay_buffer, env, optim, config, mode='train')
        scheduler.step()

        wandb.log({'train/loss_epoch': loss_train}, step=wandb.run.summary['step'])
        if wandb.run.summary['epoch'] % 10 == 0:
            torch.save(net.state_dict(), Path(wandb.run.dir) / ('model_%03d.t7' % epoch))

        checkpoint_project(net, scheduler, config)


def get_run_name(parsed):
    return '-'.join(map(str, [
        'dagger' if parsed.dagger else 'bc', parsed.method,            # paradigm
        parsed.resnet_model, 'pre' if parsed.pretrained else 'scratch' # model
        f'{parsed.proxy}2{parsed.target}',                             # modalities
        'aug' if parsed.augmentation else 'noaug',                     # dataset
        parsed.batch_size, parsed.lr, parsed.weight_decay              # hyperparams
    ])) + f'-v{parsed.description}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=Path, default='checkpoints')

    # Teacher args.
    parser.add_argument('--proxy', choices=MODELS.keys(), required=True)

    # Student args.
    parser.add_argument('--target', choices=MODALITIES, required=True)
    parser.add_argument('--resnet_model', required=True)
    parser.add_argument('--method', choices=['feedforward', 'backprop'], required=True)
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--pretrained', action='store_true')

    # Data args.
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--augmentation', action='store_true')

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()
    run_name = get_run_name(parsed)

    checkpoint_dir = parsed.checkpoint_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
            'run_name': run_name,
            'max_epoch': parsed.max_epoch,
            'checkpoint_dir': checkpoint_dir,

            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'gpu': torch.cuda.get_device_name(0),

            'teacher_args': {
                'proxy': parsed.proxy
                },

            'student_args': {
                'target': parsed.target,
                'resnet_model': parsed.resnet_model,
                'method': parsed.method,
                'dagger': parsed.dagger,
                'pretrained': parsed.pretrained
                },

            'data_args': {
                'batch_size': parsed.batch_size,
                'augmentation': parsed.augmentation
                },

            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay
                }
            }

    main(config)


