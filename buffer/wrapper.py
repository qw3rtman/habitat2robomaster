import torch
import numpy as np
from pyquaternion import Quaternion

from pathlib import Path
import random
import shutil
from collections import deque, defaultdict

import habitat
from habitat.config import Config
from habitat.config.default import get_config
from habitat.core.env import Env
from gym.spaces import Box, Dict, Discrete

import sys
sys.path.append('/u/nimit/Documents/robomaster/habitat2robomaster')
from habitat_baselines.agents.ppo_agents import PPOAgent
from habitat_dataset import HabitatDataset

TASKS = ['pointgoal']
MODES = ['student', 'teacher', 'both']
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
    def __init__(self, task, proxy, target, save='rgb', mode='teacher', student=None, shuffle=True, split='train', dataset='castle', scenes='*', gpu_id=0, sensors=['RGB_SENSOR', 'DEPTH_SENSOR'], compass=False, **kwargs):
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clean(self):
        self.agent.reset()
        self.observations = self.env.reset()

        self.i = 0

        self.mask = torch.ones(1).to(self.device)
        self.state = self.env.sim.get_agent_state()

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

        if self.compass:
            #goal = torch.as_tensor(self.observations['pointgoal_with_gps_compass'])
            r, t = self.observations['pointgoal_with_gps_compass']
            goal = torch.FloatTensor([r, np.cos(-t), np.sin(-t)])
        else:
            goal = self.get_direction()
        goal = goal.unsqueeze(dim=0).to(self.device)

        out = self.student((target, goal))
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

            self.i += 1

            if self.mode == 'both': # wp 2/iter, take the expert action for 5 steps
                beta = 0.9 * (0.95**(self.epoch/5))
                if np.random.random() <= beta:
                    self.remaining_oracle += 2

                _action = action['teacher'] if self.remaining_oracle > 0 else action['student']
                self.agent.prev_actions[0, 0] = _action['action']
                self.observations = self.env.step(_action)

                self.remaining_oracle = max(0, self.remaining_oracle-1)
            elif self.mode == 'student':
                self.observations = self.env.step(action['student'])
            elif self.mode == 'teacher':
                self.observations = self.env.step(action['teacher'])


def replay_episode(env, replay_buffer, score_by=None):
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    if score_by:
        score_by.eval()

    env.clean()
    for step in env.rollout():
        target = np.uint8(step['rgb'])
        if env.compass:
            r, t = step['compass_r'], step['compass_t']
            goal = torch.FloatTensor([r, np.cos(-t), np.sin(-t)])
        else:
            goal = env.get_direction()
        action = step['action']['teacher' if env.mode == 'both' else env.mode]['action']

        loss = random.random()
        if score_by is not None:
            _target = torch.as_tensor(target, device=env.device).unsqueeze(dim=0)
            _goal = torch.as_tensor(goal, device=env.device).unsqueeze(dim=0)
            __action = score_by((_target, _goal)).logits
            _action = torch.as_tensor([action], device=env.device)

            loss = criterion(__action, _action).item()

        replay_buffer.insert(target, goal, 0, action, loss=loss)
