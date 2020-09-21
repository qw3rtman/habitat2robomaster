import torch
import torch.nn as nn
import numpy as np

from gym import spaces

from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.common.utils import CategoricalNet

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class PointGoalPolicy(nn.Module):
    def __init__(self, resnet_model='resnet50', resnet_baseplanes=32, hidden_size=128, action_dim=3, goal_dim=3, **kwargs):
        super().__init__()

        """
        observation_spaces = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8)
        })
        """

        observation_spaces = spaces.Dict({
            'semantic': spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.uint8)
        })

        self.visual_encoder = ResNetEncoder(
            observation_spaces,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes//2,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs='rgb' in observation_spaces.spaces,
            input_channels=1)#3) # NOTE: change depending on input!!

        self.visual_fc = nn.Sequential(
            Flatten(),
            #nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
            nn.Linear(2304, hidden_size), # hack
            nn.ReLU(True))

        self.goal_fc = nn.Linear(goal_dim, hidden_size)

        self.concat_fc = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size//2))

        self.action_distribution = CategoricalNet(hidden_size//2, action_dim)

    def forward(self, rgb, goal):
        visual_feats = self.visual_encoder({'semantic': rgb})
        visual_encoding = self.visual_fc(visual_feats)
        goal_encoding = self.goal_fc(goal)

        features = torch.cat([visual_encoding, goal_encoding], dim=1)

        return self.action_distribution(self.concat_fc(features))

class InverseDynamics(nn.Module):
    def __init__(self, target, resnet_model='resnet50', resnet_baseplanes=32, hidden_size=512, dim_actions=4, height=256, width=256, **kwargs):
        super().__init__()

        self.target = target
        assert self.target in MODALITIES
        if self.target == 'depth':
            input_channels = 1
            target_space = spaces.Box(low=0, high=1, shape=(height, width, input_channels), dtype=np.float32)
        elif self.target == 'rgb':
            input_channels = 3
            target_space = spaces.Box(low=0, high=255, shape=(height, width, input_channels), dtype=np.uint8)
        elif self.target == 'semantic':
            input_channels = C
            target_space = spaces.Box(low=0, high=1, shape=(height, width, input_channels), dtype=np.bool)

        observation_spaces = spaces.Dict({self.target: target_space})

        self.t1 = ResNetEncoder(
            observation_spaces,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes//2,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs=self.target=='rgb',
            input_channels=input_channels)

        self.t2 = ResNetEncoder(
            observation_spaces,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes//2,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs=self.target=='rgb',
            input_channels=input_channels)

        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Linear(np.prod(self.t1.output_shape)+np.prod(self.t2.output_shape), hidden_size),
            nn.ReLU(True))

        self.action_distribution = CategoricalNet(hidden_size, dim_actions)

    def forward(self, x1, x2):
        visual_feats = self.visual_fc(torch.stack([
            self.t1({self.target: x1}),
            self.t2({self.target: x2})], dim=1))

        return self.action_distribution(visual_feats)
