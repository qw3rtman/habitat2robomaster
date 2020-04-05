import torch.nn as nn
from resnet import ResnetBase
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.common.utils import CategoricalNet
from gym import spaces

def spatial_softmax_base():
    return nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(True)
    )


class DirectImitation(ResnetBase): # v2.x
    def __init__(self, resnet_model='resnet34', **resnet_kwargs):
        resnet_kwargs['input_channel'] = resnet_kwargs.get('input_channel', 3)

        super().__init__(resnet_model, **resnet_kwargs)

        self.normalize = nn.BatchNorm2d(resnet_kwargs['input_channel'])
        self.deconv = spatial_softmax_base()
        self.extract = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 5, 1, 1, 0))#,
                #common.SpatialSoftmax(temperature))
        
        self.fc1 = nn.Linear(64, 1)
        self.fc2 = nn.Linear(64, 1)

        self.softmax = nn.Softmax(dim=-1)

    # TODO: take start_pos, start_rot, end_pos
    def forward(self, x):
        rgb = x[0]

        rgb = self.normalize(rgb)
        rgb = self.conv(rgb)
        rgb = self.deconv(rgb)

        return self.softmax(self.fc2(self.fc1(self.extract(rgb)).squeeze()).squeeze())


class ConditionalImitation(ResnetBase): # 3.x
    def __init__(self, resnet_model='resnet34', **resnet_kwargs):
        resnet_kwargs['input_channel'] = resnet_kwargs.get('input_channel', 3)

        super().__init__(resnet_model, **resnet_kwargs)

        self.normalize = nn.BatchNorm2d(resnet_kwargs['input_channel'])
        self.deconv = spatial_softmax_base()
        self.extract = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 5, 1, 1, 0))#,
                #common.SpatialSoftmax(temperature))
        
        self.fc1 = nn.Linear(64, 1)
        self.fc2 = nn.Linear(64, 2)
        self.fc3 = nn.Linear(10, 5)

        self.softmax = nn.Softmax(dim=1)

    # TODO: take start_pos, start_rot, end_pos
    def forward(self, x):
        rgb, meta = x

        rgb = self.normalize(rgb)
        rgb = self.conv(rgb)
        rgb = self.deconv(rgb)
        rgb = self.extract(rgb)
        rgb = self.fc1(rgb).squeeze()
        rgb = self.fc2(rgb)
        x = self.fc3(rgb.view((-1, 10)) + meta)

        return self.softmax(x)


class DirectImitationDDPPO(nn.Module): # v4.x
    def __init__(self, resnet_model='se_resneXt50', baseplanes=32, ngroups=16, hidden_size=512, dim_actions=4):
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
        rgb_vec = self.visual_encoder({'rgb': torch.FloatTensor(np.uint8(transform_(rgb))).unsqueeze(dim=0)})

        return self.action_distribution(self.visual_fc(rgb_vec)).logits
