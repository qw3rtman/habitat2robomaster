import wandb
import argparse
from pathlib import Path

import torch
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from .model import PointGoalPolicy, InverseDynamics, TemporalDistance, SceneLocalization, PointGoalPolicyAux
from .wrapper import Rollout
from .dataset import polar1, polar2, rff
from .buffer import ReplayBuffer

ACTIONS = ['F', 'L', 'R']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--split', required=True)
    parsed = parser.parse_args()

    config = yaml.load((parsed.model.parent / 'config.yaml').read_text())
    run_name = f"{config['run_name']['value']}-model_{parsed.epoch:03}-{parsed.split}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #aux_net = TemporalDistance(**config['aux_model_args']['value']).to(device)
    #aux_net = InverseDynamics(**config['aux_model_args']['value']).to(device)
    aux_net = SceneLocalization(**config['aux_model_args']['value']).to(device)
    #aux_net.load_state_dict(torch.load(config['aux_model']['value'], map_location=device))

    net = PointGoalPolicyAux(aux_net, **config['model_args']['value']).to(device)
    net.load_state_dict(torch.load(parsed.model, map_location=device))
    net.eval()

    # NOTE: we are finetuning this part
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optim = torch.optim.Adam(net.aux.parameters(), lr=2e-4, weight_decay=3.8e-7)

    env = Rollout(shuffle=True, split='val', dataset='replica', scenes=parsed.scene)
    replay_buffer = ReplayBuffer(parsed.scene)

    wandb.init(project='pointgoal-il-live-finetune-eval', name=run_name, config=config)
    wandb.run.summary['episode'] = 0
    wandb.run.summary['step'] = 0

    n = 100
    success, spl, softspl = np.zeros(n), np.zeros(n), np.zeros(n)
    correct, total = 0, 0
    for ep in range(n):
        print(f'start episode {ep}')
        replay_buffer.new_episode()
        images = []

        for i, step in enumerate(env.rollout(net=net, goal_fn=polar1)):
            print(f'step {i}')
            frame = Image.fromarray(step['rgb'])
            images.append(np.transpose(np.uint8(frame), (2, 0, 1)))

            #replay_buffer.insert(step['semantic'], step['action']['action'])
            replay_buffer.insert(step['rgb'], step['action']['action'])
            #if i > 0 and i % 100 == 0:

        loss_mean = None
        net.aux.train()

        #n = int(50/np.sqrt(100-ep))
        iterations = int(max(25*np.log(ep+1), 5))
        for j, (t1, t2, action, distance) in enumerate(replay_buffer.get_dataset(iterations=iterations, batch_size=min(len(replay_buffer)-1, 128), temporal_dim=4)): # NOTE: change based il or td
            print(f'train loop {j}')
            t1 = t1.to(device)
            t2 = t2.to(device)
            action = action.to(device)
            distance = distance.to(device)

            _distance = net.aux(t1, t2).logits
            loss = criterion(_distance, distance)
            #loss = criterion(_distance, action)
            loss_mean = loss.mean()

            correct += (distance == _distance.argmax(dim=1)).sum().item()
            #correct += (action == _distance.argmax(dim=1)).sum().item()
            total += t1.shape[0]

            loss_mean.backward()
            optim.step()
            optim.zero_grad()
        net.aux.eval()

        wandb.log({
            'accuracy': correct/total if total > 0 else 0,
            'loss': loss_mean.item()
        }, step=wandb.run.summary['episode'])

        metrics = env.env.get_metrics()
        success[ep] = metrics['success']
        spl[ep] = metrics['spl']
        softspl[ep] = metrics['softspl']

        log = {
            f'{parsed.scene}_video': wandb.Video(np.array(images), fps=60, format='mp4'),
            'success_mean': success[:ep+1].mean(),
            'spl_mean': spl[:ep+1].mean(),
            'softspl_mean': softspl[:ep+1].mean()
        }

        wandb.run.summary['episode'] += 1
        wandb.log(
                {('%s/%s' % ('val', k)): v for k, v in log.items()},
                step=wandb.run.summary['episode'])
