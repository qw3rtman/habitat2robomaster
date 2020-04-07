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

METRICS = ['distance_to_goal', 'success', 'spl']

jitter_threshold = {
    'rgb': 1e-2,
    'depth': 7.5e-2
}

models = {
    'rgb':   '/scratch/cluster/nimit/models/habitat/ppo/rgb.pth',
    'depth':   '/scratch/cluster/nimit/models/habitat/ppo/depth.pth',
    #'rgb':   '/Users/nimit/Documents/robomaster/habitat/models/v2/rgb.pth',
    #'depth': '/Users/nimit/Documents/robomaster/habitat/models/v2/depth.pth'
}

configs = {
    'rgb':   'rgb_test.yaml',
    'depth': 'depth_test.yaml'
}

class Rollout:
    def __init__(self, input_type, evaluate=False, model=None, dagger=False):
        c = Config()

        c.RESOLUTION       = 256
        c.HIDDEN_SIZE      = 512
        c.RANDOM_SEED      = 7

        c.PTH_GPU_ID       = 0
        c.SIMULATOR_GPU_ID = 0
        c.TORCH_GPU_ID     = 0
        c.NUM_PROCESSES    = 4

        c.INPUT_TYPE       = input_type
        c.MODEL_PATH       = models[c.INPUT_TYPE]
        c.GOAL_SENSOR_UUID = 'pointgoal_with_gps_compass'

        c.freeze()

        self.input_type = input_type
        self.dagger = dagger
        self.model = model
        self.model.eval()

        self.transform = transforms.ToTensor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.evaluate = evaluate

        env_config = get_config(configs[c.INPUT_TYPE])
        self.env = Env(config=env_config)
        self.agent = PPOAgent(c)

    def _act_custom(self, observations):
        # TODO: take `network` flag
        """
        meta = torch.cat([torch.Tensor([
            *self.env.current_episode.start_position,
            *self.env.current_episode.start_rotation,
            *self.env.current_episode.goals[0].position])])
        rgb = self.transform(observations[self.input_type]).unsqueeze(dim=0)

        meta = meta.to(self.device)
        rgb = rgb.to(self.device)

        action = {
            'action': self.model((rgb, meta)).detach().argmax().item(),
            'action_args': {}
        }
        """

        # NOTE: for DDPPO-style models
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
            # TODO: prune bad/stuck episodes as in supertux PPO
            # TODO: wall collisions, etc.
            if self.dagger:
                true_action = self.agent.act(observations)                  # supervision
                action, pred_action_logits = self._act_custom(observations) # predicted; rollout with this one
            else:
                if self.model: # custom network (i.e: student)
                    action, _ = self._act_custom(observations)
                else: # habitat network
                    action = self.agent.act(observations) # t

            state = self.env.sim.get_agent_state()    # t

            position  = state.position
            rotation  = state.rotation.components
            collision = self.env.get_metrics()['collisions']['is_collision'] if i > 0 else False

            d_pos.append(np.linalg.norm(position - p_pos))
            d_rot.append(np.linalg.norm(rotation - p_rot))
            d_col.append(int(collision))

            dd_pos = abs(p_d_pos - np.mean(d_pos))
            dd_rot = abs(p_d_rot - np.mean(d_rot))

            """
            if i > 10:
                print('i = {}.'.format(i))
                print('dd_pos: {}'.format(dd_pos))
                print('dd_rot: {}'.format(dd_rot))
                print(np.sqrt(dd_pos**2 + dd_rot**2))
                print()
            """

            # measure velocity and jitter
            is_stuck = i > 10 and ((np.max(d_pos) == 0 and np.max(d_rot) == 0) or np.sqrt(dd_pos**2 + dd_rot**2) < jitter_threshold[self.input_type])
            if is_stuck:
                #print('STUCK')
                if not self.evaluate:
                    break
            is_slide = i > 5 and np.min(d_col) == 1
            if is_slide:
                #print('COLLIDE')
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
                #'metrics': self.env.get_metrics()
            }

            p_d_pos = np.mean(d_pos)
            p_d_rot = np.mean(d_rot)

            observations = self.env.step(action)
            i += 1

def rollout_episode(env):
    steps = list()
    for step in env.rollout():
        steps.append(step)

    return steps

def get_episode(env, episode_dir, evaluate=False, incomplete_ok=False):
    """
    evaluate      : not generating episodes, evaluating some trained policy. no saving
    incomplete_ok : get incomplete episodes; use to train lane-following via DAgger efficiently
    """
    if evaluate:
        rollout_episode(env)
        return

    steps = list()
    while len(steps) < 30 or not bool(env.env.get_metrics()['success']):
        steps = rollout_episode(env)
        if incomplete_ok:
            #print(len(steps), bool(env.env.get_metrics()['success']))
            break

        #print('TRY AGAIN')
        #print(len(steps), bool(env.env.get_metrics()['success']))
        #print()

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
    parser = argparse.ArgumentParser()
    # TODO: take model_path and config_path
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--num_episodes', type=int, required=True)
    parser.add_argument('--input_type', choices=models.keys(), required=True)
    parser.add_argument('--incomplete_ok', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    summary = defaultdict(float)
    summary['ep'] = 1
    if (args.dataset_dir / 'summary.csv').exists():
        summary = pd.read_csv(args.dataset_dir / 'summary.csv').iloc[0]

    env = Rollout(args.input_type, evaluate=args.evaluate)
    for ep in range(int(summary['ep']), int(summary['ep'])+args.num_episodes):
        episode_dir = args.dataset_dir / f'{ep:06}'
        shutil.rmtree(episode_dir, ignore_errors=True)
        episode_dir.mkdir(parents=True, exist_ok=True)

        get_episode(env, episode_dir, args.evaluate, incomplete_ok=args.incomplete_ok)

        print(f'[!] finish ep {ep:06}')
        for m, v in env.env.get_metrics().items():
            if m in METRICS:
                summary[m] += v

        summary['ep'] = ep+1
        print({k: v / ep for k, v in summary.items() if k in METRICS})
        print()

        pd.DataFrame([summary]).to_csv(args.dataset_dir / 'summary.csv', index=False)
