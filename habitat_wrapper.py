import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from torchvision import transforms

from pathlib import Path
import argparse
import time
import shutil
from collections import deque, defaultdict
import quaternion

import habitat
from habitat.config import Config
from habitat.config.default import get_config
from habitat.core.env import Env
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.agents.ppo_agents import PPOAgent
from gym.spaces import Box, Dict, Discrete

DEBUG = False

TASKS = ['dontcrash', 'pointgoal', 'objectgoal']
METRICS = ['distance_to_goal', 'success', 'spl']

jitter_threshold = {
    'rgb': 1e-2,
    'depth': 7.5e-2
}

MODELS = {
    'rgb':   '/scratch/cluster/nimit/models/habitat/ppo/rgb.pth',
    'depth':   '/scratch/cluster/nimit/models/habitat/ppo/depth.pth',
    #'rgb':   '/Users/nimit/Documents/robomaster/habitat/models/v2/rgb.pth',
    #'depth': '/Users/nimit/Documents/robomaster/habitat/models/v2/depth.pth'
}

CONFIGS = {
    'rgb':   'rgb_test.yaml',
    'depth': 'depth_test.yaml'
}

class Rollout:
    def __init__(self, task, proxy, model=None, dagger=False, max_episode_length=200):
        """
        model:  evaluate via this model policy, if not None
        dagger: evaluate both PPOAgent and model at each step
        """
        assert task in TASKS

        c = Config()

        c.RESOLUTION       = 256
        c.HIDDEN_SIZE      = 512
        c.RANDOM_SEED      = 7

        c.PTH_GPU_ID       = 0
        c.SIMULATOR_GPU_ID = 0
        c.TORCH_GPU_ID     = 0
        c.NUM_PROCESSES    = 4

        c.INPUT_TYPE       = proxy
        c.MODEL_PATH       = MODELS[c.INPUT_TYPE]
        c.GOAL_SENSOR_UUID = 'pointgoal_with_gps_compass'

        c.freeze()

        self.task = task
        self.proxy = proxy
        self.dagger = dagger
        self.model = model
        self.max_episode_length = max_episode_length

        self.transform = transforms.ToTensor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        env_config = get_config(CONFIGS[c.INPUT_TYPE])
        self.env = Env(config=env_config)
        self.agent = PPOAgent(c)

        self.train()

    # evaluating rollout driven by self.model
    def eval(self):
        self.evaluate = True
        if self.model:
            self.model.eval()

    # generating data to train self.model
    def train(self):
        self.evaluate = False
        if self.model:
            self.model.eval()

    def act_custom(self, observations):
        # NOTE: get CONDITIONAL from 11682f9
        rgb = torch.Tensor(np.uint8(observations['rgb'])).unsqueeze(dim=0)
        rgb = rgb.to(self.device)

        out = self.model((rgb,))
        return {'action': out[0].argmax().item()}, out # action, logits

    def rollout(self):
        self.agent.reset()
        observations = self.env.reset()

        i = 0

        d_pos = deque(maxlen=10)
        d_rot = deque(maxlen=10)
        d_col = deque(maxlen=5)

        p_d_pos = 0
        p_d_rot = 0

        state = self.env.sim.get_agent_state()
        p_pos = state.position
        p_rot = state.rotation.components
        p_col = self.env.get_metrics()['collisions']['is_collision'] if i > 0 else False

        while not self.env.episode_over:
            if self.dagger:
                true_action = self.agent.act(observations)                 # supervision
                action, pred_action_logits = self.act_custom(observations) # predicted; rollout with this one
            else:
                if self.model: # custom network (i.e: student)
                    action, _ = self.act_custom(observations)
                else: # habitat network
                    action = self.agent.act(observations)

            # NOTE: stop command
            if self.task == 'dontcrash' and action['action'] == 0:
                break

            state = self.env.sim.get_agent_state()

            position  = state.position
            rotation  = state.rotation.components
            collision = self.env.get_metrics()['collisions']['is_collision'] if i > 0 else False

            d_pos.append(np.linalg.norm(position - p_pos))
            d_rot.append(np.linalg.norm(rotation - p_rot))
            d_col.append(int(collision))

            dd_pos = abs(p_d_pos - np.mean(d_pos))
            dd_rot = abs(p_d_rot - np.mean(d_rot))

            # measure velocity and jitter
            is_stuck = i > 10 and ((np.max(d_pos) == 0 and np.max(d_rot) == 0) or np.sqrt(dd_pos**2 + dd_rot**2) < jitter_threshold[self.proxy])
            if is_stuck:
                if not self.evaluate:
                    break

            is_slide = i > 5 and np.min(d_col) == 1
            if is_slide:
                if not self.evaluate:
                    break

            yield {
                'step': i,
                'action': action['action'], # what we took
                'pred_action_logits': pred_action_logits if self.dagger else None, # predicted logits
                'true_action': true_action['action'] if self.dagger else None,
                'position': position,
                'rotation': rotation,
                'collision': collision,
                'rgb': observations['rgb'],
                'depth': observations['depth'],
                #'semantic': observations['semantic'],
                'is_stuck': is_stuck,
                'is_slide': is_slide
            }

            p_d_pos = np.mean(d_pos)
            p_d_rot = np.mean(d_rot)

            i += 1
            if i >= self.max_episode_length and not self.evaluate:
                break

            observations = self.env.step(action)

# NOTE: storing in a list takes a lot of memory. keeps rgb on CUDA from rollout method
def rollout_episode(env):
    steps = list()
    for step in env.rollout():
        steps.append(step)

    return steps

def get_episode(env, episode_dir):
    if env.evaluate:
        rollout_episode(env)
        return

    steps = list()
    while len(steps) < 30 or not bool(env.env.get_metrics()['success']):
        steps = rollout_episode(env)
        if DEBUG:
            print('TRY AGAIN')
            print(len(steps), bool(env.env.get_metrics()['success']))
            print()

        if env.task == 'dontcrash': # truly Markovian
            break

    stats = list()
    for i, step in enumerate(steps):
        Image.fromarray(step['rgb']).save(episode_dir / f'rgb_{i:04}.png')
        #np.save(episode_dir / f'seg_{i:04}', step['semantic'])

        stats.append({
            'step': step['step'],
            'action': step['action'],
            'collision': step['collision'],
            'x': step['position'][0],
            'y': step['position'][1],
            'z': step['position'][2],
            'i': step['rotation'][0],
            'j': step['rotation'][1],
            'k': step['rotation'][2],
            'l': step['rotation'][3]
        })

    pd.DataFrame(stats).to_csv(episode_dir / 'episode.csv', index=False)

    info = env.env.get_metrics()
    info['collisions'] = info['collisions']['count']
    info['start_pos_x'], info['start_pos_y'], info['start_pos_z']                      = env.env.current_episode.start_position
    info['start_rot_i'], info['start_rot_j'], info['start_rot_k'], info['start_rot_l'] = env.env.current_episode.start_rotation
    info['end_pos_x'], info['end_pos_y'], info['end_pos_z']                            = env.env.current_episode.goals[0].position
    pd.DataFrame([info]).to_csv(episode_dir / 'info.csv', index=False)

if __name__ == '__main__':
    DEBUG = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--num_episodes', type=int, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--proxy', choices=MODELS.keys(), required=True)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    summary = defaultdict(float)
    summary['ep'] = 1
    if (args.dataset_dir / 'summary.csv').exists():
        summary = pd.read_csv(args.dataset_dir / 'summary.csv').iloc[0]

    env = Rollout(args.task, args.proxy)
    if args.evaluate:
        env.eval()

    for ep in range(int(summary['ep']), int(summary['ep'])+args.num_episodes):
        episode_dir = args.dataset_dir / f'{ep:06}'
        shutil.rmtree(episode_dir, ignore_errors=True)
        episode_dir.mkdir(parents=True, exist_ok=True)

        get_episode(env, episode_dir)

        print(f'[!] finish ep {ep:06}')
        for m, v in env.env.get_metrics().items():
            if m in METRICS:
                summary[m] += v

        summary['ep'] = ep+1
        print({k: v / ep for k, v in summary.items() if k in METRICS})
        print()

        pd.DataFrame([summary]).to_csv(args.dataset_dir / 'summary.csv', index=False)
