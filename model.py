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
        return ConditionalImitation(**resnet_kwargs)
    
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

        self.action_fc = nn.Linear(hidden_size, dim_actions)

        #nn.init.orthogonal_(self.action_fc.weight, gain=0.01)
        #nn.init.constant_(self.action_fc.bias, 0)

    def forward(self, x):
        rgb = x[0]
        rgb_vec = self.visual_encoder({'rgb': rgb})

        return self.action_fc(self.visual_fc(rgb_vec))

class ConditionalImitation(DirectImitation):
    def __init__(self, resnet_model='resnet18', baseplanes=32, ngroups=16, hidden_size=512, dim_actions=4, meta_size=2):
        super().__init__(resnet_model, baseplanes, ngroups, hidden_size, dim_actions)

        meta_embedding_size = hidden_size // 16
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_size, meta_embedding_size),
            nn.ReLU(True)
        )

        self.action_fc = nn.Linear(hidden_size + meta_embedding_size, dim_actions)

    def forward(self, x):
        rgb, meta = x
        rgb_vec = self.visual_encoder({'rgb': rgb})

        return self.action_fc(torch.cat([self.visual_fc(rgb_vec), self.meta_fc(meta)], dim=1))
