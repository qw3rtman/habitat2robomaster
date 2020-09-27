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

from habitat_baselines.common.env_utils import construct_envs
#from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class

from .dataset import make_onehot, polar1

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
    def __init__(self, shuffle=True, split='train', dataset='castle', scenes='*', gpu_id=0, k=0, **kwargs):
        self.dataset = dataset
        self.scenes = scenes

        ####### environment config ############################################
        config = get_config(CONFIGS['ddppo'])
        config.defrost()

        config['NUM_PROCESSES'] = 4

        sensors = ['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']
        for sensor in sensors:
            config['SIMULATOR'][sensor].HEIGHT = 160 # 256
            config['SIMULATOR'][sensor].WIDTH  = 384 # 256
            config['SIMULATOR'][sensor].HFOV   = 120 # 90
            config['SIMULATOR'][sensor].POSITION = [0.0, 0.25, 0.0] # 1.5

        config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = shuffle # NOTE: not working?
        config.SENSORS                              = sensors
        config.SIMULATOR.AGENT_0.SENSORS            = sensors
        config.DATASET.SPLIT                        = SPLIT[dataset][split]
        config.DATASET.CONTENT_SCENES               = [scenes]
        config.DATASET.DATA_PATH                    = DATAPATH[dataset]
        config.DATASET.SCENES_DIR                   = '/scratch/cluster/nimit/habitat/habitat-api/data/scene_datasets'

        # nonsense
        config.TASK_CONFIG = Config()
        config.TASK_CONFIG.DATASET = config.DATASET
        config.TASK_CONFIG.SIMULATOR = config.SIMULATOR
        config.TASK_CONFIG.SEED = 1337
        config.SIMULATOR_GPU_ID = 0

        config.freeze()

        #self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self.env = Env(config=config)
        #######################################################################

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.greedy_follower = ShortestPathFollower(self.env.sim, 0.2, False)

    def clean(self):
        self.i = 0
        self.observations = self.env.reset()
        self.state = self.env.sim.get_agent_state()

    def act(self, net=None, goal_fn=polar1):
        if net is not None:
            #rgb = torch.as_tensor(self.observations['rgb'], dtype=torch.float, device=self.device).unsqueeze(dim=0)
            rgb = torch.as_tensor(make_onehot(np.uint8(self.observations['semantic']), scene='frl_apartment_4'), dtype=torch.float, device=self.device)
            r, t = self.observations['pointgoal_with_gps_compass']
            if np.sqrt(r**2+t**2) < 0.2: # hardcode STOP
                return {'action': 0}
            goal = goal_fn(r, t).to(self.device).reshape(1, -1)

            return { # NOTE: sampling, not argmax
                'action': net(rgb, goal).sample().item() + 1
            }
        else:
            try:
                greedy_action = self.greedy_follower.get_next_action(self.env.current_episode.goals[0].position)
            except: # GreedyFollowerError, rare but it happens
                return None

            return {
                'action': 0 if greedy_action is None else greedy_action
            }

    def step(self, action):
        self.observations = self.env.step(action)

    def rollout(self, expose=True, net=None, goal_fn=polar1):
        self.clean()

        while not self.env.episode_over:
            action = self.act(net=net, goal_fn=goal_fn)
            if action is None:
                break

            if expose:
                self.state = self.env.sim.get_agent_state()

                sample = {
                    'step': self.i,
                    'action': action,
                    'position': self.state.position,
                    'rotation': self.state.rotation.components,
                    'rgb': self.observations['rgb'],
                    'depth': self.observations['depth'],
                    'semantic': self.observations['semantic'],
                    'compass_r': self.observations['pointgoal_with_gps_compass'][0],
                    'compass_t': self.observations['pointgoal_with_gps_compass'][1]
                }

                yield sample
            else:
                yield None

            self.step(action)
            self.i += 1


def save_episode(env, episode_dir):
    rgb, depth, semantic, stats = [], [], [], []
    for i, step in enumerate(env.rollout()):
        rgb.append(step['rgb'])
        depth.append(step['depth'])
        semantic.append(step['semantic'])

        stats.append({
            'action': step['action']['action'],
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

    z = zarr.open(str(episode_dir / 'rgb'), mode='w', shape=(len(rgb), *rgb[0].shape), chunks=False, dtype='u1', compressor=compressor)
    z[:] = np.array(rgb)

    #z = zarr.open(str(episode_dir / 'depth'), mode='w', shape=(len(depth), *depth[0].shape), chunks=False, dtype='f', compressor=compressor)
    #z[:] = np.array(depth)

    z = zarr.open(str(episode_dir / 'semantic'), mode='w', shape=(len(semantic), *semantic[0].shape), chunks=False, dtype='u1', compressor=compressor)
    z[:] = np.array(semantic)
