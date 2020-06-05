import argparse
from pathlib import Path
import time
import tqdm
import pickle
import wandb

import torch
import torchvision
import numpy as np

import sys
sys.path.append('/u/nimit/Documents/robomaster/habitat2robomaster')

from model import GoalConditioned
from wrapper import MODELS, MODALITIES, SPLIT, Rollout, replay_episode
from frame_buffer import ReplayBuffer, LossSampler
from util import C, make_onehot


def loop(net, data, replay_buffer, uids, env, optim, config, mode='train'):
    if mode == 'train':
        net.train()
    else:
        net.eval()

    losses = list()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    correct, total = 0, 0
    tick = time.time()
    for idx, target, goal, _, action in tqdm.tqdm(data, desc=mode, total=len(data), leave=False):
        if config['student_args']['target'] == 'semantic':
            target = make_onehot(target.reshape(-1, config['data_args']['height'],
                config['data_args']['width'])).to(config['device'])
        else:
            target = torch.as_tensor(target, device=config['device'], dtype=torch.float32)
            target = target.reshape(config['data_args']['batch_size'],
                    config['data_args']['height'], config['data_args']['width'], -1)

        goal = torch.as_tensor(goal, device=config['device'], dtype=torch.float32)
        action = torch.as_tensor(action, device=config['device'], dtype=torch.int64)

        if config['data_args']['augmentation']:
            # stop augmentation
            probs = np.random.random(config['data_args']['batch_size'])
            goal[probs < 0.05, 0] = 0.2 * torch.rand(sum(probs < 0.05), device=config['device'])
            action[probs < 0.05]  = 0

            """ flip augmentation
            probs = np.random.random(config['data_args']['batch_size'])
            target[probs < 0.05] = target[probs < 0.05].flip(dims=(2,))
            goal[probs < 0.05]   *= torch.tensor([1, 1, -1], device=config['device']) # -theta
            left, right = action[probs < 0.05] == 2, action[probs < 0.05] == 3
            action[probs < 0.05][left], action[probs < 0.05][right] = 3, 2
            """

        print(target.shape)
        _action = net((target, goal)).logits
        loss = criterion(_action, action)
        loss_mean = loss.mean()

        uids.update(replay_buffer.uids[idx])
        correct += (action == _action.argmax(dim=1)).sum().item()
        total += target.shape[0]

        if mode == 'train':
            loss_mean.backward()
            optim.step()
            optim.zero_grad()

        #replay_buffer.update_loss(idx, loss.detach().cpu().numpy())
        losses.append(loss_mean.item())

        wandb.run.summary['step'] += 1
        metrics = {'loss': loss_mean.item(),
                   'images_per_second': target.shape[0] / (time.time() - tick),
                   'unique_samples': len(uids)}
        wandb.log({('%s/%s' % ('train', k)): v for k, v in metrics.items()},
                   step=wandb.run.summary['step'])
        tick = time.time()

    wandb.log({'train/acc': correct/total}, step=wandb.run.summary['step'])
    return np.mean(losses)


def resume_project(net, scheduler, replay_buffer, uids, config):
    print('Resumed at epoch %d.' % wandb.run.summary['epoch'])

    net.load_state_dict(torch.load(config['checkpoint_dir'] / 'model_latest.t7'))
    scheduler.load_state_dict(torch.load(config['checkpoint_dir'] / 'scheduler_latest.t7'))
    if (config['checkpoint_dir'] / 'buffer').exists():
        replay_buffer.load(config['checkpoint_dir'] / 'buffer')
    with open(config['checkpoint_dir'] / 'uids.pickle', 'rb') as f:
        uids = pickle.load(f)

def checkpoint_project(net, scheduler, replay_buffer, uids, config):
    torch.save(net.state_dict(), config['checkpoint_dir'] / 'model_latest.t7')
    torch.save(scheduler.state_dict(), config['checkpoint_dir'] / 'scheduler_latest.t7')
    replay_buffer.save(config['checkpoint_dir'] / 'buffer', overwrite=True)
    with open(config['checkpoint_dir'] / 'uids.pickle', 'wb') as f:
        pickle.dump(uids, f)


def main(config):
    net = GoalConditioned(**config['student_args'], **config['data_args']).to(config['device'])
    optim = torch.optim.Adam(net.parameters(), **config['optimizer_args'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=0.5,
            milestones=[config['max_epoch'] * 0.5, config['max_epoch'] * 0.75])

    # TODO: should support gibson+mp3d, not just gibson
    sensors = ['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']
    mode = 'teacher' if config['teacher_args']['supervision'] == 'ddppo' else config['teacher_args']['supervision']
    env = Rollout('pointgoal', config['teacher_args']['proxy'],
            config['student_args']['target'], mode=mode, shuffle=False,
            split='train', dataset=config['teacher_args']['dataset'],
            sensors=sensors[:(3 if config['student_args']['target'] == 'semantic' else 2)],
            k=3 if config['student_args']['dagger'] else 0, **config['data_args'])

    uids = set()
    replay_buffer = ReplayBuffer(int(2e5), history_size=int(config['student_args']['history_size']),
            dshape=(config['data_args']['height'], config['data_args']['width'],
                3 if config['student_args']['target'] == 'rgb' else 1),
            dtype=torch.uint8, goal_size=config['student_args']['goal_size'])
    starter_buffer = Path('/scratch/cluster/nimit/data/habitat/%s-%s2%s-buffer' % \
            (config['teacher_args']['dataset'], config['teacher_args']['proxy'], config['student_args']['target']))
    starter = False
    if starter_buffer.exists():
        replay_buffer.load(starter_buffer)
        starter = True

    sampler = LossSampler(replay_buffer, config['data_args']['batch_size'])

    project_name = 'habitat-pointgoal-{}-student'.format(config['teacher_args']['proxy'])
    wandb.init(project=project_name, config=config, id=config['run_name'], resume='auto')
    wandb.save(str(Path(wandb.run.dir) / '*.t7'))
    if wandb.run.resumed:
        resume_project(net, scheduler, replay_buffer, uids, config)
    else:
        wandb.run.summary['step'] = 0
        wandb.run.summary['epoch'] = 0

    for epoch in tqdm.tqdm(range(wandb.run.summary['epoch']+1, config['max_epoch']+1), desc='epoch'):
        wandb.run.summary['epoch'] = epoch

        if not starter:
            env.env._episode_iterator.max_scene_repeat_episodes = 4 if epoch == 1 else 16
            for _ in range(4 if epoch == 1 else 512):#512 if epoch > 1 else 1328):
                replay_episode(env, replay_buffer)#, score_by=net)
                print(env.env.get_metrics())

        dataset = replay_buffer.get_dataset()
        data = torch.utils.data.DataLoader(dataset, num_workers=0, pin_memory=True, batch_sampler=sampler)

        loss_train = loop(net, data, replay_buffer, uids, env, optim, config, mode='train')
        scheduler.step()

        # 4 episodes per scenes to populate the buffer; then 32 scenes per epoch
        if starter and epoch > 1:
            env.env._episode_iterator.max_scene_repeat_episodes = 16
            for _ in range(256):#512 if epoch > 1 else 1328):
                replay_episode(env, replay_buffer)#, score_by=net)

        wandb.log({'train/loss_epoch': loss_train, 'buffer_capacity': replay_buffer.size}, step=wandb.run.summary['step'])
        if wandb.run.summary['epoch'] % 10 == 0:
            torch.save(net.state_dict(), Path(wandb.run.dir) / ('model_%03d.t7' % epoch))

        checkpoint_project(net, scheduler, replay_buffer, uids, config)


def get_run_name(parsed):
    return '-'.join(map(str, [
        'dagger' if parsed.dagger else 'bc', parsed.method,                  # paradigm
        parsed.hidden_size, parsed.resnet_model,                             # model
        parsed.supervision, 'pre' if parsed.pretrained else 'scratch',       # model
        parsed.dataset, f'{parsed.proxy}2{parsed.target}',                   # modalities
        parsed.history_size, parsed.goal,                                    # dataset
        'aug' if parsed.augmentation else 'noaug',                           # dataset
        f'{parsed.height}x{parsed.width}', parsed.fov, parsed.camera_height, # dataset
        parsed.batch_size, parsed.lr, parsed.weight_decay                    # hyperparams
    ])) + f'-v{parsed.description}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=Path, default='checkpoints')

    # Teacher args.
    parser.add_argument('--supervision', choices=['ddppo', 'greedy'], required=True)
    parser.add_argument('--proxy', choices=MODELS.keys(), required=True)
    parser.add_argument('--dataset', choices=SPLIT.keys(), required=True)

    # Student args.
    parser.add_argument('--target', choices=MODALITIES, required=True)
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--resnet_model', required=True)
    parser.add_argument('--history_size', type=int, default=1)
    parser.add_argument('--method', choices=['feedforward', 'backprop'], required=True)
    parser.add_argument('--goal', choices=['polar', 'cartesian'], required=True)
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--pretrained', action='store_true')

    # Data args.
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--fov', type=int, default=90)
    parser.add_argument('--camera_height', type=float, default=1.5)

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
                'supervision': parsed.supervision,
                'proxy': parsed.proxy,
                'dataset': parsed.dataset
                },

            'student_args': {
                'goal_size': 3 if parsed.goal == 'polar' else 2,
                'target': parsed.target,
                'resnet_model': parsed.resnet_model,
                'history_size': parsed.history_size,
                'hidden_size': parsed.hidden_size,
                'method': parsed.method,
                'dagger': parsed.dagger,
                'pretrained': parsed.pretrained
                },

            'data_args': {
                'batch_size': parsed.batch_size,
                'augmentation': parsed.augmentation,
                'height': parsed.height,
                'width': parsed.width,
                'fov': parsed.fov,
                'camera_height': parsed.camera_height
                },

            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay
                }
            }

    main(config)


