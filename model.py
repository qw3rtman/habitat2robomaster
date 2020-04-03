import torch.nn as nn
from resnet import ResnetBase

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


class Network(ResnetBase):
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

        self.softmax = nn.Softmax(dim=1)

    # TODO: take start_pos, start_rot, end_pos
    def forward(self, x):
        x = self.normalize(x)
        x = self.conv(x)
        x = self.deconv(x)

        return self.softmax(self.fc2(self.fc1(self.extract(x)).squeeze()).squeeze())

if __name__ == '__main__':
    net = Network(resnet_model='resnet34')
