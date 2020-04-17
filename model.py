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


class ConditionalStateEncoderImitation(nn.Module):
    def __init__(self, resnet_model='resnet50', batch_size=64):
        super(ConditionalStateEncoderImitation, self).__init__()

        observation_spaces, action_spaces = spaces.Dict({
            # 'depth': spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.float32),
            'rgb': spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
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
            tgt_mode='nimit' # NOTE: (dx, dy)
        )

        self.test_recurrent_hidden_states = torch.zeros(self.actor_critic.net.num_recurrent_layers, batch_size, 512) # move to GPU
        self.not_done_masks = torch.zeros(batch_size, 1) # move to GPU
        self.prev_actions = torch.zeros(batch_size, 1, dtype=torch.long) # move to GPU

    def forward(self, x):
        rgb, meta = x
        batch = {'rgb': rgb, 'pointgoal_with_gps_compass': meta}

        _, actions, action_log_probs, self.test_recurrent_hidden_states = self.actor_critic.act(
            batch,
            self.test_recurrent_hidden_states.detach(),
            self.prev_actions,
            self.not_done_masks,
            deterministic=False)

        self.prev_actions.copy_(actions)
        # NOTE: Make masks not done till reset (end of episode) will be called
        # NOTE: have some dataset __getitem__ property that triggers this?
        self.not_done_masks = torch.ones(1, 1) # move to GPU

        return self.actor_critic.prev_distribution.logits
