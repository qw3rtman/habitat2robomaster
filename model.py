import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from gym import spaces

from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.common.utils import CategoricalNet
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy

from habitat_wrapper import MODALITIES
from buffer.util import C

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def get_model(target, rnn=False, **resnet_kwargs):
    if rnn:
        return ConditionalStateEncoderImitation(target, **resnet_kwargs)

    return GoalConditioned(target, **resnet_kwargs)

class GoalConditioned(nn.Module):
    def __init__(self, target, resnet_model='resnet50', resnet_baseplanes=32, hidden_size=512, history_size=1, dim_actions=4, goal_size=3, **kwargs):
        super().__init__()

        self.target = target
        assert self.target in MODALITIES
        if self.target == 'depth':
            input_channels = 1
            target_space = spaces.Box(low=0, high=1, shape=(256, 256, history_size), dtype=np.float32)
        elif self.target == 'rgb':
            input_channels = 3
            target_space = spaces.Box(low=0, high=255, shape=(256, 256, 3*history_size), dtype=np.uint8)
        elif self.target == 'semantic':
            input_channels = C
            target_space = spaces.Box(low=0, high=1, shape=(256, 256, C*history_size), dtype=np.bool)

        observation_spaces = spaces.Dict({self.target: target_space})

        self.visual_encoder = ResNetEncoder(
            observation_spaces,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes//2,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs=(self.target=='rgb'),
            input_channels=input_channels*history_size)

        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
            nn.ReLU(True))

        self.goal_fc = nn.Linear(goal_size, 32)
        self.action_distribution = CategoricalNet(hidden_size+32, dim_actions)

    def forward(self, x):
        target, goal = x
        visual_feats = self.visual_fc(self.visual_encoder({self.target: target}))
        goal_encoding = self.goal_fc(goal)

        features = torch.cat([visual_feats, goal_encoding], dim=1)
        return self.action_distribution(features)


class ConditionalStateEncoderImitation(nn.Module):
    def __init__(self, target, batch_size, resnet_model='resnet50', tgt_mode='ddppo', **kwargs):
        super(ConditionalStateEncoderImitation, self).__init__()

        self.target = target
        assert self.target in MODALITIES
        if self.target == 'semantic':
            self.target = 'depth'

        if self.target == 'depth':
            input_channels = 1
            target_space = spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.float32)
        elif self.target == 'rgb':
            input_channels = 3
            target_space = spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8)

        self.observation_spaces, self.action_spaces = spaces.Dict({
            self.target: target_space,
            'pointgoal_with_gps_compass': spaces.Box( 
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
        }), spaces.Discrete(4)

        print(tgt_mode)
        self.actor_critic = PointNavResNetPolicy(
            observation_space=self.observation_spaces,
            action_space=self.action_spaces,
            hidden_size=512,
            rnn_type='LSTM',
            num_recurrent_layers=2,
            backbone=resnet_model,
            goal_sensor_uuid='pointgoal_with_gps_compass',
            normalize_visual_inputs='rgb' in self.observation_spaces.spaces.keys(),
            #tgt_mode='nimit', # NOTE: (dx, dy)
            tgt_mode=tgt_mode, # NOTE: direct pointgoal_with_gps_compass inputs
            input_channels=input_channels
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        self.value = torch.Tensor([0.0])
        self.hidden_states = torch.zeros(self.actor_critic.net.num_recurrent_layers, self.batch_size, 512).to(self.device)

    def clean(self):
        self.hidden_states = torch.zeros(self.actor_critic.net.num_recurrent_layers, self.batch_size, 512).to(self.device)

    def forward(self, x):
        #assert x[0].shape[0] == self.batch_size
        # B x ...
        target, direction, prev_action, mask = x
        batch = {self.target: target, 'pointgoal_with_gps_compass': direction}

        self.value, _, _, self.hidden_states = self.actor_critic.act(
            batch,
            self.hidden_states[:,:self.batch_size], # NOTE: no BPTT
            prev_action.unsqueeze(dim=1)[:self.batch_size],
            mask.unsqueeze(dim=1)[:self.batch_size],
            deterministic=False)

        return self.actor_critic.prev_distribution.logits
