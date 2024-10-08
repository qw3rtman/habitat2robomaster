from itertools import chain

import torch
import torch.nn as nn
import numpy as np

from gym import spaces

from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.common.utils import CategoricalNet

from .const import GIBSON_IDX2NAME, HEIGHT, WIDTH
from .util import SpatialSoftmax


observation_spaces = spaces.Dict({
    'rgb': spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
})

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PointGoalPolicy(nn.Module):
    def __init__(self, resnet_model='resnet50', resnet_baseplanes=32, hidden_size=128, action_dim=3, goal_dim=3, **kwargs):
        super().__init__()

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
    def __init__(self, resnet_model='resnet50', resnet_baseplanes=32, hidden_size=512, localization_dim=2, scene_bias=True, **kwargs):
        super().__init__()
        self.localization_dim = localization_dim

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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, localization_dim))

        self.scene_xy_fc = nn.ModuleDict({
            name: nn.Linear(2, 2, bias=True) for name in GIBSON_IDX2NAME
        })

        self.scene_t_fc= nn.ModuleDict({
            name: nn.Linear(2, 2, bias=False) for name in GIBSON_IDX2NAME
        })

    def forward(self, rgb, scene_idx):
        visual_features = self.visual_fc1(self.R1({'rgb': rgb}))
        shared_features = self.localization_fc(visual_features)

        out = torch.empty((rgb.shape[0], self.localization_dim)).cuda()
        for s in scene_idx.long().unique():
            xy_affine = self.scene_xy_fc[GIBSON_IDX2NAME[s]]
            t_rotate = self.scene_t_fc[GIBSON_IDX2NAME[s]]
            out[scene_idx==s] = torch.cat([
                xy_affine(shared_features[scene_idx==s,:2]).view(-1, 2),
                t_rotate(shared_features[scene_idx==s,2:]).view(-1, 2)
            ], dim=1)

        return out

class GoalPrediction(nn.Module):
    def __init__(self, resnet_model='resnet50', resnet_baseplanes=32, hidden_size=512, steps=5, temperature=1.0, **kwargs):
        super().__init__()
        self.steps = steps

        self.R1 = ResNetEncoder(
            observation_spaces,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes//2,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs='rgb' in observation_spaces.spaces,
            input_channels=3 if 'rgb' in observation_spaces.spaces else 1)

        self.visual_fc1 = nn.Sequential(
            nn.Linear(self.R1.output_shape[0], hidden_size),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(*chain(*[
            (
                nn.GroupNorm(dim, dim),
                nn.ConvTranspose2d(dim, dim//2, 3, 2, 1, 1),
                nn.ReLU(True)
            ) for dim in hidden_size//(2**np.arange(3))
        ]))

        self.extract = nn.ModuleList([
            nn.Sequential(
                nn.GrouNorm(hidden_size//8, hidden_size//8),
                nn.Conv2d(hidden_size//8, steps, 1, 1, 0),
                SpatialSoftmax(temperature)
            ) for i in range(3)
        ])

    def forward(self, rgb, action, scene_idx):
        visual_features = self.visual_fc1(self.R1({'rgb': rgb}).permute(0, 2, 3, 1))
        shared_features = self.deconv(visual_features.permute(0, 3, 1, 2))

        out = torch.empty((action.shape[0], self.steps, 2)).cuda()
        for a in action.long().unique():
            out[action==a] = self.extract[a](shared_features[action==a])

        return out


class PointGoalPolicyAux(nn.Module): # Auxiliary task
    def __init__(self, aux_model, hidden_size=128, action_dim=3, goal_dim=3, **kwargs):
        super().__init__()

        self.aux = aux_model

        self.goal_fc = nn.Linear(goal_dim, hidden_size)

        self.concat_fc = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size//2))

        self.action_distribution = CategoricalNet(hidden_size//2, action_dim)

    def forward(self, rgb, goal):
        visual_features = self.aux.visual_fc1(self.aux.R1({'rgb': rgb}))
        shared_features = self.aux.localization_fc[:1](visual_features)

        features = torch.cat([
            shared_features,
            self.goal_fc(goal)
        ], dim=1)

        return self.action_distribution(self.concat_fc(features))
