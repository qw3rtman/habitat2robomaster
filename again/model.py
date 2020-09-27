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
    def __init__(self, resnet_model='resnet50', resnet_baseplanes=32, hidden_size=512, action_dim=3, **kwargs):
        super().__init__()

        observation_spaces = spaces.Dict({
            'semantic': spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.uint8)
        })

        self.R1 = ResNetEncoder(
            observation_spaces,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes//2,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs=False,
            input_channels=1)

        self.visual_fc1 = nn.Sequential(
            Flatten(),
            #nn.Linear(np.prod(self.R1.output_shape), hidden_size),
            nn.Linear(2304, hidden_size), # hack
            nn.ReLU(True))

        """
        self.R2 = ResNetEncoder(
            observation_spaces,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes//2,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs=False,
            input_channels=1)

        self.visual_fc2 = nn.Sequential(
            Flatten(),
            #nn.Linear(np.prod(self.R2.output_shape), hidden_size),
            nn.Linear(2304, hidden_size), # hack
            nn.ReLU(True))
        """

        self.concat_fc = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size//2))

        self.action_distribution = CategoricalNet(hidden_size//2, action_dim)

    def forward(self, rgb1, rgb2):
        return self.action_distribution(self.concat_fc(torch.cat([
            self.visual_fc1(self.R1({'semantic': rgb1})),
            self.visual_fc1(self.R1({'semantic': rgb2}))
        ], dim=1)))

class PointGoalPolicyAux(nn.Module): # Auxiliary task of Inverse Dynamics
    def __init__(self, inverse_dynamics, resnet_model='resnet50', resnet_baseplanes=32, hidden_size=128, action_dim=3, goal_dim=3, **kwargs):
        super().__init__()

        self.id = inverse_dynamics

        self.goal_fc = nn.Linear(goal_dim, hidden_size)

        self.concat_fc = nn.Sequential(
            nn.Linear(3*hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size//2))

        self.action_distribution = CategoricalNet(hidden_size//2, action_dim)

    def forward(self, rgb, goal):
        # TODO: freeze self.id weights
        features = torch.cat([
            self.id.visual_fc1(self.id.R1({'semantic': rgb})),
            self.goal_fc(goal)
        ], dim=1)

        return self.action_distribution(self.concat_fc(features))
