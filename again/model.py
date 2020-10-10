import torch
import torch.nn as nn
import numpy as np

from gym import spaces

from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.common.utils import CategoricalNet

from .pointgoal_dataset import HEIGHT, WIDTH
from .const import GIBSON_IDX2NAME

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class PointGoalPolicy(nn.Module):
    def __init__(self, resnet_model='resnet50', resnet_baseplanes=32, hidden_size=128, action_dim=3, goal_dim=3, **kwargs):
        super().__init__()

        """
        observation_spaces = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        })
        """

        observation_spaces = spaces.Dict({
            'semantic': spaces.Box(low=0, high=1, shape=(HEIGHT, WIDTH, 1), dtype=np.uint8)
        })

        self.visual_encoder = ResNetEncoder(
            observation_spaces,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes//2,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs='rgb' in observation_spaces.spaces,
            input_channels=3 if 'rgb' in observation_spaces.spaces else 1)

        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
            #nn.Linear(2304, hidden_size), # hack
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

        """
        observation_spaces = spaces.Dict({
            'semantic': spaces.Box(low=0, high=1, shape=(HEIGHT, WIDTH, 1), dtype=np.uint8)
        })
        """

        observation_spaces = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        })

        self.R1 = ResNetEncoder(
            observation_spaces,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes//2,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs='rgb' in observation_spaces.spaces,
            input_channels=3 if 'rgb' in observation_spaces.spaces else 1)

        self.visual_fc1 = nn.Sequential(
            Flatten(),
            nn.Linear(np.prod(self.R1.output_shape), hidden_size),
            #nn.Linear(2304, hidden_size), # hack
            nn.ReLU(True))

        self.concat_fc = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size//2))

        self.action_distribution = CategoricalNet(hidden_size//2, action_dim)

    def forward(self, rgb1, rgb2):
        return self.action_distribution(self.concat_fc(torch.cat([
            self.visual_fc1(self.R1({'rgb': rgb1})),
            self.visual_fc1(self.R1({'rgb': rgb2}))
        ], dim=1)))#.logits

class TemporalDistance(nn.Module):
    def __init__(self, resnet_model='resnet50', resnet_baseplanes=32, hidden_size=512, temporal_dim=3, **kwargs):
        super().__init__()

        """
        observation_spaces = spaces.Dict({
            'semantic': spaces.Box(low=0, high=1, shape=(HEIGHT, WIDTH, 1), dtype=np.uint8)
        })
        """

        observation_spaces = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        })

        self.R1 = ResNetEncoder(
            observation_spaces,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes//2,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs='rgb' in observation_spaces.spaces,
            input_channels=3 if 'rgb' in observation_spaces.spaces else 1)

        self.visual_fc1 = nn.Sequential(
            Flatten(),
            nn.Linear(np.prod(self.R1.output_shape), hidden_size),
            #nn.Linear(2304, hidden_size), # hack
            nn.ReLU(True))

        self.concat_fc = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size//2))

        self.temporal_distribution = CategoricalNet(hidden_size//2, temporal_dim)

    def forward(self, rgb1, rgb2):
        return self.temporal_distribution(self.concat_fc(torch.cat([
            self.visual_fc1(self.R1({'rgb': rgb1})),
            self.visual_fc1(self.R1({'rgb': rgb2}))
        ], dim=1)))

class SceneLocalization(nn.Module):
    def __init__(self, resnet_model='resnet50', resnet_baseplanes=32, hidden_size=512, scene_bias=True, **kwargs):
        super().__init__()

        observation_spaces = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        })

        self.R1 = ResNetEncoder(
            observation_spaces,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes//2,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs='rgb' in observation_spaces.spaces,
            input_channels=3 if 'rgb' in observation_spaces.spaces else 1)

        self.visual_fc1 = nn.Sequential(
            Flatten(),
            #nn.Linear(np.prod(self.R1.output_shape), hidden_size),
            nn.Linear(7680, hidden_size), # hack
            nn.ReLU(True))

        self.localization_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(True))

        self.scene_fc = nn.ModuleDict({
            name: nn.Linear(hidden_size//2, 2, bias=scene_bias) for name in GIBSON_IDX2NAME
        })

    def forward(self, rgb, scene_idx):
        visual_features = self.visual_fc1(self.R1({'rgb': rgb}))
        shared_features = self.localization_fc(visual_features)

        out = torch.empty((rgb.shape[0], 2)).cuda()
        for s in scene_idx.long().unique():
            extract = self.scene_fc[GIBSON_IDX2NAME[s]]
            out[scene_idx==s] = extract(shared_features[scene_idx==s])

        return out

class PointGoalPolicyAux(nn.Module): # Auxiliary task
    def __init__(self, aux_model, hidden_size=128, action_dim=3, goal_dim=3, **kwargs):
        super().__init__()

        self.aux = aux_model

        self.goal_fc = nn.Linear(goal_dim, hidden_size//2)

        self.concat_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size//2))

        self.action_distribution = CategoricalNet(hidden_size//2, action_dim)

    def forward(self, rgb, goal):
        features = torch.cat([
            self.aux.localization_fc(self.aux.visual_fc1(self.aux.R1({'rgb': rgb}))),
            self.goal_fc(goal)
        ], dim=1)

        return self.action_distribution(self.concat_fc(features))
