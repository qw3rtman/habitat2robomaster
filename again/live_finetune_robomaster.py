import wandb
import argparse
from pathlib import Path
import pickle

import torch
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from .model import PointGoalPolicy, InverseDynamics, TemporalDistance, PointGoalPolicyAux
from .dataset import polar1, polar2, rff
from .buffer import ReplayBuffer

from ipaddress import ip_address
import pygame

import sys
sys.path.append('/Users/nimit/Documents/robomaster/low_level_control/robomaster_control')
from agents.habitat import HabitatAgent
from envs.robomaster_env import RobomasterEnv

ACTIONS = ['F', 'L', 'R']

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--buffer', type=Path)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--ip', type=ip_address)
    parser.add_argument('--port', type=int, default=10607)
    parsed = parser.parse_args()

    config = yaml.load((parsed.model.parent / 'config.yaml').read_text())
    run_name = f"{config['run_name']['value']}-model_{parsed.epoch:03}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #aux_net = TemporalDistance(**config['aux_model_args']['value']).to(device)
    aux_net = InverseDynamics(**config['aux_model_args']['value']).to(device)
    #aux_net.load_state_dict(torch.load(config['aux_model']['value'], map_location=device))

    net = PointGoalPolicyAux(aux_net, **config['model_args']['value']).to(device)
    net.load_state_dict(torch.load(parsed.model, map_location=device))
    net.eval()

    # NOTE: we are finetuning this part
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optim = torch.optim.Adam(net.aux.parameters(), lr=2e-4, weight_decay=3.8e-7)

    if parsed.buffer is not None:
        with open(parsed.buffer, 'rb') as f:
            replay_buffer = pickle.load(f)
    else:
        replay_buffer = ReplayBuffer('apartment_0')

    wandb.init(project='pointgoal-il-live-finetune-eval', name=run_name, config=config)
    wandb.run.summary['episode'] = 0
    wandb.run.summary['step'] = 0

    GOAL = np.array([1.0, 0.0])

    agent = HabitatAgent()
    screen = pygame.display.set_mode((1280 // 2, 720 // 2))

    n = 5
    correct, total = 0, 0
    for ep in range(n):
        env = RobomasterEnv(parsed.ip, parsed.port)

        print(f'start episode {ep}')
        replay_buffer.new_episode()
        images = []

        state = env.step()
        action = agent.act(state)

        while True:
        #for i in range(24):

            screen.blit(pygame.surfarray.make_surface(state['image'].transpose(1,0,2)), (0,0))
            pygame.display.update()

            location = np.array([state['x'], -state['y']])
            if np.linalg.norm(GOAL-location) < 0.3:
                break

            r, t  = cart2pol(*(GOAL-location))
            print(GOAL-location)
            print(r, t)
            print()
            goal = polar1(r, t).to(device).reshape(1, -1)
            image = Image.fromarray(state['image']).resize((384, 160))

            rgb = torch.as_tensor(np.array(image), dtype=torch.float, device=device).unsqueeze(dim=0)

            _action = net(rgb, goal).sample().item()
            action = agent.act(state, action=_action)

            replay_buffer.insert(np.array(image), _action+1)
            images.append(np.transpose(np.uint8(image), (2, 0, 1)))

            state = env.step(action, render=True)

        loss_mean = None
        net.aux.train()

        print(len(replay_buffer))

        #n = int(50/np.sqrt(100-ep))
        iterations = 8 #int(max(25*np.log(ep+1), 5))
        for j, (t1, t2, action, distance) in enumerate(replay_buffer.get_dataset(iterations=iterations, batch_size=16, temporal_dim=1)): # NOTE: change based il or td
            print(f'train loop {j}')
            t1 = t1.to(device)
            t2 = t2.to(device)
            action = action.to(device)
            distance = distance.to(device)

            _distance = net.aux(t1, t2).logits
            #loss = criterion(_distance, distance)
            loss = criterion(_distance, action)
            loss_mean = loss.mean()

            #correct += (distance == _distance.argmax(dim=1)).sum().item()
            correct += (action == _distance.argmax(dim=1)).sum().item()
            total += t1.shape[0]

            loss_mean.backward()
            optim.step()
            optim.zero_grad()
        net.aux.eval()

        torch.save(net.state_dict(), 'model_latest.t7')
        with open('buffer_latest.pickle', 'wb') as f:
            pickle.dump(replay_buffer, f)

        wandb.log({
            'accuracy': correct/total if total > 0 else 0,
            'loss': loss_mean.item()
        }, step=wandb.run.summary['episode'])

        log = {
            f'video': wandb.Video(np.array(images), fps=20, format='mp4')
        }

        wandb.run.summary['episode'] += 1
        wandb.log(
                {('%s/%s' % ('val', k)): v for k, v in log.items()},
                step=wandb.run.summary['episode'])

        env.clean()
