import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from gym import spaces

from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.common.utils import CategoricalNet

from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy
from habitat_baselines.common.utils import batch_obs

from habitat_wrapper import MODALITIES

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def get_model(target, conditional=False, rnn=False, input_channels=3, **resnet_kwargs):
    if rnn:
        return ConditionalStateEncoderImitation(target, input_channels=input_channels, **resnet_kwargs)

    if conditional:
        return ConditionalImitation(target, **resnet_kwargs)
    
    return DirectImitation(target, **resnet_kwargs)


class DirectImitation(nn.Module):
    def __init__(self, target, resnet_model='resnet50', baseplanes=32, ngroups=16, hidden_size=512, dim_actions=4):
        super().__init__()

        self.target = target
        assert self.target in MODALITIES
        if self.target == 'semantic':
            self.target = 'depth'

        if self.target == 'depth':
            target_space = spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.float32)
        elif self.target == 'rgb':
            target_space = spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8)

        observation_spaces = spaces.Dict({self.target: target_space})

        self.visual_encoder = ResNetEncoder(
            observation_spaces,
            baseplanes=baseplanes,
            ngroups=ngroups,
            make_backbone=getattr(resnet, resnet_model),
            normalize_visual_inputs=(self.target=='rgb')
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
        target = x[0]
        visual_embedding = self.visual_encoder({self.target: target})

        return self.action_fc(self.visual_fc(visual_embedding))

class ConditionalImitation(DirectImitation):
    def __init__(self, target, resnet_model='resnet50', baseplanes=32, ngroups=16, hidden_size=512, dim_actions=4, meta_size=2, **kwargs):
        super().__init__(target, resnet_model, baseplanes, ngroups, hidden_size, dim_actions)

        meta_embedding_size = hidden_size // 16
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_size, meta_embedding_size),
            nn.ReLU(True)
        )

        self.action_fc = nn.Linear(hidden_size + meta_embedding_size, dim_actions)

    def forward(self, x):
        target, meta = x
        visual_embedding = self.visual_encoder({self.target: target})

        return self.action_fc(torch.cat([self.visual_fc(visual_embedding), self.meta_fc(meta)], dim=1))


class ConditionalStateEncoderImitation(nn.Module):
    def __init__(self, target, batch_size, resnet_model='resnet50', input_channels=3, **kwargs):
        super(ConditionalStateEncoderImitation, self).__init__()

        self.target = target
        assert self.target in MODALITIES
        if self.target == 'semantic':
            self.target = 'depth'

        if self.target == 'depth':
            target_space = spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.float32)
        elif self.target == 'rgb':
            target_space = spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8)

        observation_spaces, action_spaces = spaces.Dict({
            self.target: target_space,
            'pointgoal_with_gps_compass': spaces.Box( 
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
        }), spaces.Discrete(4)

        self.actor_critic = PointNavResNetPolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=512,
            rnn_type='LSTM',
            num_recurrent_layers=2,
            backbone=resnet_model,
            goal_sensor_uuid='pointgoal_with_gps_compass',
            normalize_visual_inputs='rgb' in observation_spaces.spaces.keys(),
            tgt_mode='nimit', # NOTE: (dx, dy)
            input_channels=input_channels
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        self.value = torch.Tensor([0.0])
        self.hidden_states = torch.zeros(self.actor_critic.net.num_recurrent_layers, self.batch_size, 512).to(self.device)

    def clean(self):
        self.hidden_states = torch.zeros(self.actor_critic.net.num_recurrent_layers, self.batch_size, 512).to(self.device)

    def forward(self, x):
        assert x[0].shape[0] == self.batch_size
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
