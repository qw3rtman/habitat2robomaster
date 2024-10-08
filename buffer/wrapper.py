import torch
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from numcodecs import Blosc
import zarr

from pathlib import Path
import random
import shutil
from collections import deque, defaultdict
from operator import itemgetter

import habitat
from habitat.config import Config
from habitat.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.core.env import Env
from gym.spaces import Box, Dict, Discrete

import sys
sys.path.append('/u/nimit/Documents/robomaster/visual-navigation')
from habitat_baselines.agents.ppo_agents import PPOAgent
from habitat_dataset import HabitatDataset
from util import C, make_onehot, draw_top_down_map

TASKS = ['pointgoal']
MODES = ['student', 'teacher', 'both', 'greedy']
METRICS = ['success', 'spl', 'softspl', 'distance_to_goal']
MODALITIES = ['depth', 'rgb', 'semantic']

MODELS = {
    'depth': {
        'ddppo': ('/scratch/cluster/nimit/models/habitat/ddppo/gibson-4plus-mp3d-train-val-test-resnet50.pth', 'resnet50'),
    },
    'rgb': {
        'ddppo': ('/scratch/cluster/nimit/models/habitat/ddppo/gibson-2plus-mp3d-train-val-test-se-resneXt50-rgb.pth', 'se_resneXt50'),
    }
}

CONFIGS = {
    'ddppo':      'configs/pointgoal/ddppo/val.yaml'
}

DATAPATH = {
    'castle':     '/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/castle/{split}/{split}.json.gz',
    'office':     '/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz',
    'replica':     '/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/replica/v1/{split}/{split}.json.gz',
    'mp3d':       '/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz',
    'gibson':     '/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz'
}

SPLIT = {
    'castle':    {'train': 'train', 'val': 'val'},
    'office':    {'train': 'B6ByNegPMKs_train', 'val': 'B6ByNegPMKs_val'},
    'replica':   {'train': 'train', 'val': 'val', 'test': 'test'},
    'mp3d':      {'train': 'train', 'val': 'val', 'test': 'test'},
    'gibson':    {'train': 'train', 'val': 'val', 'val_mini': 'val_mini',
                  'train_ddppo': 'train_ddppo', 'val_ddppo': 'val_ddppo'}
}

class Rollout:
    def __init__(self, task, proxy, target, mode='teacher', student=None, shuffle=True, split='train', dataset='castle', scenes='*', gpu_id=0, sensors=['RGB_SENSOR', 'DEPTH_SENSOR'], height=256, width=256, fov=90, camera_height=1.5, goal='polar', k=0, **kwargs):
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
        self.dataset = dataset
        self.scenes = scenes
        self.height = height
        self.width = width
        self.fov = fov
        self.camera_height = camera_height
        self.mode = mode
        self.student = student
        self.epoch = 1
        self.goal = goal
        self.k = k

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
        self.agent = PPOAgent(agent_config, resnet_model=resnet_model, ddppo=True)
        #######################################################################

        ####### environment config ############################################
        env_config = get_config(CONFIGS['ddppo'])
        env_config.defrost()

        for sensor in sensors:
            if sensor == 'DEPTH_SENSOR':
                env_config['SIMULATOR'][sensor].POSITION = [0., 1.5, 0.]
                continue
            env_config['SIMULATOR'][sensor].HEIGHT = self.height
            env_config['SIMULATOR'][sensor].WIDTH  = self.width
            env_config['SIMULATOR'][sensor].HFOV   = self.fov
            env_config['SIMULATOR'][sensor].POSITION = [0.0, self.camera_height, 0.0]
            # overhead
            #env_config['SIMULATOR'][sensor].POSITION = [0.0, 5, 0.0] #self.camera_height, 0.0]
            #env_config['SIMULATOR'][sensor].ORIENTATION = [-1.571, 0.0, 0.0]
        print(env_config['SIMULATOR']['SEMANTIC_SENSOR'])
        env_config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = shuffle # NOTE: not working?
        env_config.SIMULATOR.AGENT_0.SENSORS            = sensors
        env_config.DATASET.SPLIT                        = SPLIT[dataset][split]
        env_config.DATASET.CONTENT_SCENES               = [scenes]
        env_config.DATASET.DATA_PATH                    = DATAPATH[dataset]
        env_config.DATASET.SCENES_DIR                   = '/scratch/cluster/nimit/habitat/habitat-api/data/scene_datasets'

        env_config.freeze()
        self.env = Env(config=env_config) # scene, etc.
        #######################################################################

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.mode == 'greedy':
            self.greedy_follower = ShortestPathFollower(self.env.sim, 0.2, False)

    def clean(self):
        self.i = 0
        self.agent.reset()
        self.observations = self.env.reset()
        self.state = self.env.sim.get_agent_state()
        if self.student != None and hasattr(self.student, 'clean'):
            self.student.clean()

    def get_direction(self):
        source_position = self.state.position
        rot = self.state.rotation.components
        source_rotation = Quaternion(*rot[1:4], rot[0])
        goal_position = self.env.current_episode.goals[0].position
        return HabitatDataset.get_direction(source_position, source_rotation, goal_position)
    
    def get_target(self, target=None):
        _target = target or self.target
        # H x W x C
        target = self.observations[_target]
        if _target == 'semantic':
            target = target[..., np.newaxis]
        return target

    def act(self):
        def act_random():
            return {'action': int(1+(random.random()*3))}
        def act_student():
            target = self.get_target()
            if self.target == 'semantic':
                scene = self.scenes if self.dataset == 'replica' else None
                print(scene)
                target = make_onehot(np.uint8(target.reshape(-1, self.height, self.width)), scene).to(self.device)
            else:
                target = torch.as_tensor(target, dtype=torch.float, device=self.device).unsqueeze(dim=0)

            # TODO: base on parsed.goal type
            r, t = self.observations['pointgoal_with_gps_compass']
            goal = torch.as_tensor([r, np.cos(-t), np.sin(-t)], dtype=torch.float, device=self.device).unsqueeze(dim=0)

            out = self.student((target, goal))
            action = out.sample().item()

            return {'action': action}, out # action, logits

        if self.mode == 'student':
            student_action, _ = act_student()
            return {'student': student_action}
        elif self.mode == 'teacher':
            teacher_action = self.agent.act(self.observations)
            return {'teacher': teacher_action, 'random': act_random()}
        elif self.mode == 'both':
            student_action, student_logits = act_student()
            teacher_action = self.agent.act(self.observations)
            return {'student': student_action,
                    'teacher': teacher_action,
                    'student_logits': student_logits}
        elif self.mode == 'greedy':
            try:
                greedy_action = self.greedy_follower.get_next_action(self.env.current_episode.goals[0].position)
            except: # GreedyFollowerError, rare but it happens
                return None

            return {'greedy': {'action': 0 if greedy_action is None else greedy_action},
                    'random': act_random()}


    def step(self, action):
        if self.mode == 'student':
            _action = action['student']
        elif self.mode in ['teacher', 'greedy']:
            _action = action[self.mode]
            if self.i % 10 < self.k:
                _action = action['random']
        elif self.mode == 'both': # wp 2/iter, take the expert action for 5 steps
            beta = 0.9 * (0.95**(self.epoch/5))
            _action = action['teacher'] if np.random.random() <= beta else action['student']
            self.agent.prev_actions[0, 0] = _action['action'] # for teacher in mode=both

        self.observations = self.env.step(_action)

    def rollout(self, expose=True):
        self.clean()

        while not self.env.episode_over:
            action = self.act()
            if action is None:
                break

            if expose:
                self.state = self.env.sim.get_agent_state()

                sample = {
                    'step': self.i,
                    'action': action,
                    'position': self.state.position,
                    'rotation': self.state.rotation.components,
                    'rgb': self.observations['rgb'] if 'rgb' in self.observations else None,
                    'depth': self.observations['depth'] if 'depth' in self.observations else None,
                    'semantic': np.uint8(self.observations['semantic']) if 'semantic' in self.observations else None,
                    'compass_r': self.observations['pointgoal_with_gps_compass'][0],
                    'compass_t': self.observations['pointgoal_with_gps_compass'][1]
                }

                yield sample
            else:
                yield None

            self.step(action)
            self.i += 1


# TODO: handle prev_action, if needed
def replay_episode(env, replay_buffer, score_by=None):
    hsize = replay_buffer.history_size

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    if score_by:
        score_by.eval()

    target_buffer = np.empty((hsize,*replay_buffer.dshape), dtype=np.uint8)
    for i, step in enumerate(env.rollout()):
        target = env.get_target()
        r, t = step['compass_r'], step['compass_t']
        goal = torch.as_tensor([r, np.cos(-t), np.sin(-t)], dtype=torch.float)
        action = step['action']['teacher' if env.mode == 'both' else env.mode]['action']

        #np.save(f'/scratch/cluster/nimit/data/htest/{i:03}.npy', step['semantic'])

        loss = random.random()
        if score_by is not None and i >= hsize:
            _target = torch.as_tensor(target_buffer, device=env.device).unsqueeze(dim=0)
            _goal = torch.as_tensor(goal, device=env.device).unsqueeze(dim=0)
            __action = score_by((_target, _goal)).logits
            _action = torch.as_tensor([action], device=env.device)

            loss = criterion(__action, _action).item()

        if i >= hsize:
            target_buffer[:-1] = target_buffer[1:]
            target_buffer[-1] = target
            replay_buffer.insert(target_buffer, goal, 0, action, loss=loss)
        else:
            target_buffer[i] = target

    return itemgetter('success', 'spl', 'softspl')(env.env.get_metrics())


def save_episode(env, episode_dir, save=[]):
    stats, targets = [], {target: [] for target in set([env.target] + save)}
    for i, step in enumerate(env.rollout()):
        for target in targets.keys():
            targets[target].append(env.get_target(target=target))

        stats.append({
            'action': step['action'][env.mode]['action'],
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
    pd.DataFrame(stats).to_csv(episode_dir / 'episode.csv', index=False)

    compressor = Blosc(cname='zstd', clevel=3)
    for t, target in targets.items():
        z = zarr.open(str(episode_dir / t), mode='w', shape=(len(target), *target[0].shape), chunks=False, dtype='u1', compressor=compressor)
        z[:] = np.array(target)

if __name__ == '__main__':
    from PIL import Image

    env = Rollout('pointgoal', 'semantic', 'rgb', mode='greedy', dataset='replica', scene='frl_apartment_4', sensors=['RGB_SENSOR', 'SEMANTIC_SENSOR'], height=160, width=384)
    for i, step in enumerate(env.rollout()):
        Image.fromarray(step['rgb']).save(f'cam/rgb_{i:03}.png')
        np.save(f'cam/sem_{i:03}.npy', step['semantic'])
        """
        tmap = env.env.get_metrics()['top_down_map']
        if tmap is not None:
            #top_down_map = draw_top_down_map(env.env.get_metrics(), env.observations["heading"], env.env.get_metrics()['top_down_map']['map'].shape[0])
            #print(top_down_map, type(top_down_map))
            print(np.unique(tmap['map']), tmap['map'].dtype)
            np.save(f'maptest/{i:03}.npy', tmap['map'])
            #Image.fromarray(tmap['map']).save(f'maptest/{i:03}.png')
            #print(tmap['agent_map_coord'])
            #print(tmap['agent_angle'])
        """
