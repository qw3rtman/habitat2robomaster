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

TASKS = ['dontcrash', 'pointgoal', 'objectgoal']
MODES = ['student', 'teacher', 'both']
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

def get_rollout(task, proxy, student=None):
    assert task in TASKS

    if task == 'dontcrash':
        return Rollout(proxy, 'teacher', student=student)

class Rollout:
    def __init__(self, proxy, mode, student=None):
        assert proxy in MODELS.keys()
        assert mode in MODES
        if mode in ['student', 'both']:
            assert student is not None

        self.proxy = proxy
        self.mode = mode
        self.student = student

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

        self.transform = transforms.ToTensor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        env_config = get_config(CONFIGS[c.INPUT_TYPE])
        self.env = Env(config=env_config)
        self.agent = PPOAgent(c)

    def act_student(self):
        # NOTE: get CONDITIONAL from 11682f9
        rgb = torch.Tensor(np.uint8(self.observations['rgb'])).unsqueeze(dim=0)
        rgb = rgb.to(self.device)

        out = self.student((rgb,))
        return {'action': out[0].argmax().item()}, out # action, logits

    def clean(self):
        self.agent.reset()
        self.observations = self.env.reset()

        self.i = 0

        self.d_pos = deque(maxlen=10)
        self.d_rot = deque(maxlen=10)
        self.d_col = deque(maxlen=5)

        self.prev_d_pos = 0
        self.prev_d_rot = 0

        state = self.env.sim.get_agent_state()
        self.prev_pos = state.position
        self.prev_rot = state.rotation.components

    def is_slide(self):
        collision = self.env.get_metrics()['collisions']['is_collision'] if self.i > 0 else False
        self.d_col.append(int(collision))

        return self.i > 5 and np.min(self.d_col) == 1

    # measure velocity and jitter
    def is_stuck(self):
        # curr
        self.d_pos.append(np.linalg.norm(self.state.position - self.prev_pos))
        self.d_rot.append(np.linalg.norm(self.state.rotation.components - self.prev_rot))

        dd_pos = abs(self.prev_d_pos - np.mean(self.d_pos))
        dd_rot = abs(self.prev_d_rot - np.mean(self.d_rot))

        # post
        #self.prev_pos = self.state.position
        #self.prev_rot = self.state.rotation.components

        self.prev_d_pos = np.mean(self.d_pos)
        self.prev_d_rot = np.mean(self.d_rot)

        return self.i > 10 and ((np.max(self.d_pos) == 0 and np.max(self.d_rot) == 0) or np.sqrt(dd_pos**2 + dd_rot**2) < jitter_threshold[self.proxy])

    def get_action(self):
        if self.mode == 'student':
            student_action, _ = self.act_student()
            return {'student': student_action}

        if self.mode == 'teacher':
            teacher_action = self.agent.act(self.observations)
            return {'teacher': teacher_action}

        if self.mode == 'both':
            teacher_action = self.agent.act(self.observations)
            student_action, student_logits = self.act_student()
            return {
                'teacher': teacher_action,
                'student': student_action,
                'student_logits': student_logits
            }

    def rollout(self):
        self.clean()

        while not self.env.episode_over:
            action = self.get_action()
            self.state = self.env.sim.get_agent_state()

            is_stuck = self.is_stuck()
            is_slide = self.is_slide()
            sample = {
                'step': self.i,
                'action': action,
                'position': self.state.position,
                'rotation': self.state.rotation.components,
                'collision': self.env.get_metrics()['collisions']['is_collision'] if self.i > 0 else False,
                'rgb': self.observations['rgb'],
                'depth': self.observations['depth'],
                #'semantic': self.observations['semantic'],
                'is_stuck': is_stuck,
                'is_slide': is_slide
            }

            # ending early
            """
            if self.mode == 'teacher':
                if action[self.mode]['action'] == 0 or is_stuck or is_slide:
                    break
            """

            yield sample
            self.i += 1

            if self.mode == 'both': # wp 0.1, take the expert action
                self.observations = self.env.step(action['student'])# if np.random.random() > 0.05 else action['teacher'])
            else:
                self.observations = self.env.step(action[self.mode])

def rollout_episode(env):
    steps = list()
    for i, step in enumerate(env.rollout()):
        steps.append(step)
        # NOTE: storing in a list takes a lot of memory. keeps rgb on CUDA from rollout method
        if i == 200:
            break

    return steps

def get_episode(env):
    env.clean()

    #if env.mode in ['student', 'both']: # eval, DAgger
    return env.rollout()

    """
    if env.mode == 'teacher': # dataset generation
        steps = list()
        while not bool(env.env.get_metrics()['success']): #or len(steps) < 30
            steps = rollout_episode(env)

        return steps
    """

def save_episode(env, episode_dir):
    stats = list()

    lwns, longest, length = 0, 0, 0
    for i, step in enumerate(get_episode(env)):
        length += 1

        if step['is_stuck'] or step['is_slide']:
            longest = 0
            if env.mode in ['teacher', 'both']:
                continue
        longest += 1

        lwns = max(lwns, longest)

        Image.fromarray(step['rgb']).save(episode_dir / f'rgb_{i:04}.png')
        #np.save(episode_dir / f'seg_{i:04}', step['semantic'])
        if env.mode == 'both':
            action = step['action']['teacher']
        else:
            action = step['action'][env.mode]

        stats.append({
            'step': step['step'],
            'action': action['action'],
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
    info['lwns'] = lwns
    info['lwns_norm'] = lwns / length
    info['collisions'] = info['collisions']['count'] if info['collisions'] else 0
    info['start_pos_x'], info['start_pos_y'], info['start_pos_z']                      = env.env.current_episode.start_position
    info['start_rot_i'], info['start_rot_j'], info['start_rot_k'], info['start_rot_l'] = env.env.current_episode.start_rotation
    info['end_pos_x'], info['end_pos_y'], info['end_pos_z']                            = env.env.current_episode.goals[0].position
    pd.DataFrame([info]).to_csv(episode_dir / 'info.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--num_episodes', type=int, required=True)
    #parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--proxy', choices=MODELS.keys(), required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    args = parser.parse_args()

    summary = defaultdict(float)
    summary['ep'] = 1
    if (args.dataset_dir / 'summary.csv').exists():
        summary = pd.read_csv(args.dataset_dir / 'summary.csv').iloc[0]

    env = Rollout(args.proxy, args.mode)
    for ep in range(int(summary['ep']), int(summary['ep'])+args.num_episodes):
        episode_dir = args.dataset_dir / f'{ep:06}'
        shutil.rmtree(episode_dir, ignore_errors=True)
        episode_dir.mkdir(parents=True, exist_ok=True)

        save_episode(env, episode_dir)

        print(f'[!] finish ep {ep:06}')
        for m, v in env.env.get_metrics().items():
            if m in METRICS:
                summary[m] += v

        summary['ep'] = ep+1
        print({k: v / ep for k, v in summary.items() if k in METRICS})
        print()

        pd.DataFrame([summary]).to_csv(args.dataset_dir / 'summary.csv', index=False)
