import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from torchvision import transforms
from pyquaternion import Quaternion
from numcodecs import Blosc
import zarr

from pathlib import Path
import random
import argparse
import time
import shutil
from collections import deque, defaultdict

from habitat_dataset import HabitatDataset

import habitat
from habitat.config import Config
from habitat.config.default import get_config
from habitat.core.env import Env
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.agents.ppo_agents import PPOAgent
from gym.spaces import Box, Dict, Discrete

TASKS = ['dontcrash', 'pointgoal', 'objectgoal']
MODES = ['student', 'teacher', 'both']
METRICS = ['success', 'spl', 'softspl', 'distance_to_goal']
MODALITIES = ['depth', 'rgb', 'semantic']

jitter_threshold = {
    'rgb': 1e-2,
    'depth': 7.5e-2,
    'semantic': 7.5e-2
}

MODELS = {
    'depth': {
        'ppo':   ('/scratch/cluster/nimit/models/habitat/ppo/depth.pth', ''),
        'ddppo': ('/scratch/cluster/nimit/models/habitat/ddppo/gibson-4plus-mp3d-train-val-test-resnet50.pth', 'resnet50'),
    },
    'rgb': {
        'ppo':   ('/scratch/cluster/nimit/models/habitat/ppo/rgb.pth', ''),
        'ddppo': ('/scratch/cluster/nimit/models/habitat/ddppo/gibson-2plus-mp3d-train-val-test-se-resneXt50-rgb.pth', 'se_resneXt50'),
    }
}

CONFIGS = {
    'ppo':        'configs/pointgoal/ppo/val.yaml',
    'ddppo':      'configs/pointgoal/ddppo/val.yaml'
}

DATAPATH = {
    'castle':     '/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/castle/{split}/{split}.json.gz',
    'office':     '/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz',
    'mp3d':       '/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz',
    'gibson':     '/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz'
}

SPLIT = {
    'castle':    {'train': 'train', 'val': 'val'},
    'office':    {'train': 'B6ByNegPMKs_train', 'val': 'B6ByNegPMKs_val'},
    'mp3d':      {'train': 'train', 'val': 'val', 'test': 'test'},
    'gibson':    {'train': 'train', 'val': 'val', 'val_mini': 'val_mini',
                  'train_ddppo': 'train_ddppo', 'val_ddppo': 'val_ddppo'}
}

class Rollout:
    def __init__(self, task, proxy, target, save='rgb', mode='teacher', student=None, rnn=False, shuffle=True, split='train', dataset='castle', scenes='*', gpu_id=0, sensors=['RGB_SENSOR', 'DEPTH_SENSOR'], compass=False, **kwargs):
        assert task in TASKS
        assert proxy in MODALITIES
        assert target in MODALITIES
        assert mode in MODES
        assert dataset in DATAPATH.keys()
        assert dataset in SPLIT.keys()
        assert split in SPLIT[dataset]
        if mode in ['student', 'both']:
            assert student is not None

        self.task = task
        self.proxy = proxy
        self.target = target
        self.save = save
        self.mode = mode
        self.student = student
        self.epoch = 1
        self.rnn = rnn
        self.compass = compass

        ####### agent config ##################################################
        agent_config = Config()
        agent_config.INPUT_TYPE               = proxy
        agent_config.RESOLUTION               = 256
        agent_config.HIDDEN_SIZE              = 512
        agent_config.GOAL_SENSOR_UUID         = 'pointgoal_with_gps_compass'

        agent_config.MODEL_PATH, resnet_model = '', 'resnet50'
        if mode in ['teacher', 'both']:
            agent_config.MODEL_PATH, resnet_model = MODELS[agent_config.INPUT_TYPE]['ddppo']

        agent_config.RANDOM_SEED              = 7
        agent_config.PTH_GPU_ID               = gpu_id
        agent_config.SIMULATOR_GPU_ID         = gpu_id
        agent_config.TORCH_GPU_ID             = gpu_id
        agent_config.NUM_PROCESSES            = 8

        agent_config.freeze()
        self.agent = PPOAgent(agent_config, ddppo=True, resnet_model=resnet_model)
        #######################################################################

        ####### environment config ############################################
        env_config = get_config(CONFIGS['ddppo'])
        env_config.defrost()

        env_config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = shuffle # NOTE: not working?
        env_config.SIMULATOR.AGENT_0.SENSORS            = sensors
        env_config.DATASET.SPLIT                        = SPLIT[dataset][split]
        env_config.DATASET.CONTENT_SCENES               = [scenes]
        env_config.DATASET.DATA_PATH                    = DATAPATH[dataset]
        env_config.DATASET.SCENES_DIR                   = '/scratch/cluster/nimit/habitat/habitat-api/data/scene_datasets'

        env_config.freeze()
        self.env = Env(config=env_config) # scene, etc.
        #######################################################################

        self.remaining_oracle = 0
        self.transform = transforms.ToTensor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clean(self):
        self.agent.reset()
        self.observations = self.env.reset()

        self.i = 0

        self.prev_action = torch.zeros(1, dtype=torch.long).to(self.device)
        self.mask = torch.ones(1).to(self.device)

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

    def get_direction(self):
        source_position = self.state.position
        rot = self.state.rotation.components
        source_rotation = Quaternion(*rot[1:4], rot[0])
        goal_position = self.env.current_episode.goals[0].position
        return HabitatDataset.get_direction(source_position, source_rotation, goal_position)

    def act_student(self):
        if self.target == 'depth':
            target = torch.as_tensor(self.observations[self.target]).unsqueeze(dim=0).float()
        elif self.target == 'rgb':
            target = torch.as_tensor(self.observations[self.target]).unsqueeze(dim=0).float()
        elif self.target == 'semantic':
            onehot = HabitatDataset.make_semantic(self.observations['semantic'])
            target = onehot.unsqueeze(dim=0).float()

        target = target.to(self.device)

        if self.task == 'dontcrash':
            out = self.student((target,))
        elif self.task == 'pointgoal':
            if self.compass:
                #goal = torch.as_tensor(self.observations['pointgoal_with_gps_compass'])
                r, t = self.observations['pointgoal_with_gps_compass']
                goal = torch.FloatTensor([r, np.cos(-t), np.sin(-t)])
            else:
                goal = self.get_direction()
            goal = goal.unsqueeze(dim=0).to(self.device)

            if self.rnn:
                out = self.student((target, goal, self.prev_action, self.mask))
                action = torch.distributions.Categorical(torch.softmax(out, dim=1)).sample().to(self.device).item()
            else:
                out = self.student((target, goal, self.prev_action))
                action = out.sample().item()

        return {'action': action}, out # action, logits

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

    def rollout(self, expose=True):
        self.clean()
        self.state = self.env.sim.get_agent_state()
        if self.student != None and hasattr(self.student, 'clean'):
            self.student.clean()

        self.remaining_oracle = 0

        while not self.env.episode_over:
            action = self.get_action()
            #print(action)

            if expose:
                self.state = self.env.sim.get_agent_state()

                is_stuck = self.is_stuck()
                is_slide = self.is_slide()

                sample = {
                    'step': self.i,
                    'action': action,
                    'position': self.state.position,
                    'rotation': self.state.rotation.components,
                    'collision': self.env.get_metrics()['collisions']['is_collision'] if self.i > 0 else False,
                    'rgb': self.observations['rgb'] if 'rgb' in self.observations else None,
                    'depth': self.observations['depth'] if 'depth' in self.observations else None,
                    'semantic': np.uint8(self.observations['semantic']) if 'semantic' in self.observations else None,
                    'compass_r': self.observations['pointgoal_with_gps_compass'][0],
                    'compass_t': self.observations['pointgoal_with_gps_compass'][1],
                    'is_stuck': is_stuck,
                    'is_slide': is_slide
                }

                # pruning rules
                if self.mode == 'teacher':
                    if self.task == 'dontcrash' and action[self.mode]['action'] == 0:
                        break

                yield sample
            else:
                yield None

            self.i += 1

            if self.mode == 'both': # wp 2/iter, take the expert action for 5 steps
                beta = 0.9 * (0.95**(self.epoch/5))
                if np.random.random() <= beta:
                    self.remaining_oracle += 2

                _action = action['teacher'] if self.remaining_oracle > 0 else action['student']
                self.prev_action[0] = _action['action']
                self.agent.prev_actions[0, 0] = _action['action']
                self.observations = self.env.step(_action)

                self.remaining_oracle = max(0, self.remaining_oracle-1)
            elif self.mode == 'student':
                self.prev_action[0] = action['student']['action']
                self.observations = self.env.step(action['student'])
            elif self.mode == 'teacher':
                self.agent.prev_actions[0, 0] = action['teacher']['action']
                self.observations = self.env.step(action['teacher'])

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


def replay_episode(env, replay_buffer, score_by=None):
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    if score_by:
        score_by.eval()

    prev_action = 0
    for step in get_episode(env):
        target = np.uint8(step['rgb'])
        goal = np.float32([step['compass_r'], np.cos(-step['compass_t']), np.sin(-step['compass_t'])])
        action = step['action']['teacher' if env.mode == 'both' else env.mode]['action']

        loss = random.random()
        if score_by is not None:
            _target = torch.as_tensor(target, device=env.device).unsqueeze(dim=0)
            _goal = torch.as_tensor(goal, device=env.device).unsqueeze(dim=0)
            _prev_action = torch.as_tensor(prev_action, device=env.device).unsqueeze(dim=0)
            __action = score_by((_target, _goal, _prev_action)).logits
            _action = torch.as_tensor([action], device=env.device)

            loss = criterion(__action, _action).item()

        replay_buffer.insert(target, goal, prev_action, action, loss=loss)
        prev_action = action


def save_episode(env, episode_dir, max_len=-1):
    stats = list()
    depths, rgbs, segs = [], [], []
    lwns, longest, length = 0, 0, 0

    episode = get_episode(env)
    if max_len > -1:
        episode = list(episode)
        if len(episode) >= max_len or len(episode) < 4: # don't save
            return

    for i, step in enumerate(episode):
        length += 1

        """
        if step['is_stuck'] or step['is_slide']:
            longest = 0
            if env.mode in ['teacher', 'both']:
                continue
        """
        longest += 1

        lwns = max(lwns, longest)

        if env.save=='depth' and 'depth' in step and step['depth'] is not None:
            depths.append(step['depth'])

        if env.save=='rgb' and 'rgb' in step and step['rgb'] is not None:
            rgbs.append(step['rgb'])
            #Image.fromarray(step['rgb']).save(episode_dir / f'rgb_{i:04}.png')

        if env.save=='semantic' and 'semantic' in step and step['semantic'] is not None:
            segs.append(step['semantic'])
            #np.savez_compressed(episode_dir / f'seg_{i:04}', semantic=step['semantic'])

        if env.mode == 'both':
            action = step['action']['teacher']
        else:
            action = step['action'][env.mode]

        stats.append({
            'step': step['step'],
            'action': action['action'],
            'collision': step['collision'],
            'compass_r': step['compass_r'],
            'compass_t': step['compass_t'],
            'x': step['position'][0],
            'y': step['position'][1],
            'z': step['position'][2],
            'i': step['rotation'][0],
            'j': step['rotation'][1],
            'k': step['rotation'][2],
            'l': step['rotation'][3]
        })

    episode_dir.mkdir(parents=True, exist_ok=True)
    if len(depths) > 0:
        compressor = Blosc(cname='zstd', clevel=3)
        z = zarr.open(str(episode_dir / 'depth'), mode='w', shape=(len(depths), 256, 256), chunks=False, dtype='f4', compressor=compressor)
        z[:] = np.array(depths)[:,:,:,0] # squeeze

    if len(rgbs) > 0 :
        compressor = Blosc(cname='zstd', clevel=3) # ~81.3 kb per sample
        z = zarr.open(str(episode_dir / 'rgb'), mode='w', shape=(len(rgbs), 256, 256, 3), chunks=False, dtype='u1', compressor=compressor)
        z[:] = np.array(rgbs)

    if len(segs) > 0:
        compressor = Blosc(cname='zstd', clevel=3) # ~2.85 kb per sample
        z = zarr.open(str(episode_dir / 'semantic'), mode='w', shape=(len(segs), 256, 256), chunks=False, dtype='u1', compressor=compressor)
        z[:] = np.array(segs)

    pd.DataFrame(stats).to_csv(episode_dir / 'episode.csv', index=False)

    info = env.env.get_metrics()
    info['scene'] = Path(env.env.sim._current_scene).stem
    info['lwns'] = lwns
    info['lwns_norm'] = lwns / length
    info['collisions'] = info['collisions']['count'] if info['collisions'] else 0
    info['start_pos_x'], info['start_pos_y'], info['start_pos_z']                      = env.env.current_episode.start_position
    info['start_rot_i'], info['start_rot_j'], info['start_rot_k'], info['start_rot_l'] = env.env.current_episode.start_rotation
    info['end_pos_x'], info['end_pos_y'], info['end_pos_z']                            = env.env.current_episode.goals[0].position
    pd.DataFrame([info]).to_csv(episode_dir / 'info.csv', index=False)

    return stats[-1]['step']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--num_episodes', type=int, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--proxy', choices=MODELS.keys(), required=True)
    parser.add_argument('--target', choices=MODELS.keys()) # use with student
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--dataset', choices=DATAPATH.keys(), required=True)
    parser.add_argument('--scene', default='*')
    parser.add_argument('--split', required=True)
    parser.add_argument('--depth', action='store_true')
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--semantic', action='store_true')
    parser.add_argument('--num_frames', type=int)
    parsed = parser.parse_args()

    summary = defaultdict(float)
    summary['ep'] = 1
    if (parsed.dataset_dir / 'summary.csv').exists():
        summary = pd.read_csv(parsed.dataset_dir / 'summary.csv').iloc[0]

    sensors = ['DEPTH_SENSOR']
    if parsed.rgb:
        sensors.append('RGB_SENSOR')
    if parsed.semantic:
        sensors.append('SEMANTIC_SENSOR')

    env = Rollout(parsed.task, parsed.proxy, parsed.target, mode=parsed.mode, shuffle=parsed.shuffle, split=parsed.split, dataset=parsed.dataset, scenes=[parsed.scene], sensors=sensors)

    ep = int(summary['ep'])
    ending_ep = ep + parsed.num_episodes
    total_frames = 0
    while True:
        if parsed.num_frames:
            if total_frames >= parsed.num_frames:
                break
        else:
            if ep >= ending_ep:
                break

        episode_dir = parsed.dataset_dir / f'{ep:06}'
        if parsed.scene != '*':
            episode_dir = parsed.dataset_dir / f'{parsed.scene}-{parsed.split}-{ep:06}'
        shutil.rmtree(episode_dir, ignore_errors=True)
        episode_dir.mkdir(parents=True, exist_ok=True)

        num_frames = save_episode(env, episode_dir)

        print(f'[!] finish ep {ep:06}')
        for m, v in env.env.get_metrics().items():
            if m in METRICS:
                summary[m] += v

        summary['ep'] = ep+1
        print({k: v / ep for k, v in summary.items() if k in METRICS})
        print()

        pd.DataFrame([summary]).to_csv(parsed.dataset_dir / 'summary.csv', index=False)

        total_frames += num_frames
        ep += 1
