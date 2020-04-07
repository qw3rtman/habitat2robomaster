import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from gym import spaces

from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.common.utils import CategoricalNet

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def get_model(conditional=False, **resnet_kwargs):
    if conditional:
        pass
    
    return DirectImitation(**resnet_kwargs)


class DirectImitation(nn.Module):
    def __init__(self, resnet_model='resnet18', baseplanes=32, ngroups=16, hidden_size=512, dim_actions=4):
        super().__init__()
        
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({'rgb': spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype='uint8')}),
            baseplanes=baseplanes,
            ngroups=ngroups,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs=True
        )
        
        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
            nn.ReLU(True)
        )

        self.action_distribution = CategoricalNet(hidden_size, dim_actions)

    def forward(self, x):
        rgb = x[0]
        rgb_vec = self.visual_encoder({'rgb': rgb})

        return self.action_distribution(self.visual_fc(rgb_vec)).logits
