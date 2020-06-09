import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from . import resnet


class SpatialSoftmax(torch.nn.Module):
    """
    IMPORTANT:
    i in [0, 1], where 0 is at the bottom, 1 is at the top
    j in [-1, 1]

    (  1, -1) ... (  1,   1) ... (  1, 1)
              ...            ...
    (0.5, -1) ... (0.5, 0.5) ... (0.5, 1)
              ...            ...
    (  0, -1) ... (  0, 0.5) ... (  0, 1)
    ...
    """
    def __init__(self, temperature=1.0):
        super().__init__()

        self.temperature = temperature

    def forward(self, logit):
        """
        Assumes logits is size (n, c, h, w)
        """
        flat = logit.view(logit.shape[:-2] + (-1,))
        weights = F.softmax(flat / self.temperature, dim=-1).view_as(logit)

        x = (weights.sum(-2) * torch.linspace(-1, 1, logit.shape[-1]).to(logit.device)).sum(-1)
        y = (weights.sum(-1) * torch.linspace(-1, 1, logit.shape[-2]).to(logit.device)).sum(-1)

        return torch.stack((x, y), -1)


def GoalPredictionModel(steps=5, temperature=1.0, resnet_model='resnet34', **resnet_kwargs):
    return Network(steps, temperature, resnet_model, **resnet_kwargs)


class Network(resnet.ResnetBase):
    def __init__(self, steps, temperature, resnet_model, **resnet_kwargs):
        resnet_kwargs['input_channel'] = resnet_kwargs.get('input_channel', 3)

        super().__init__(resnet_model, **resnet_kwargs)

        self.normalize = torch.nn.BatchNorm2d(resnet_kwargs['input_channel'])
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(512), # 2048 for resnet50
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), # 2048 for resnet50
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(True))

        self.extract = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64,steps,1,1,0),
                SpatialSoftmax(temperature)
            ) for i in range(4)
        ])

        # TODO: query all branches at once in final distill
        #       just return all four actions at one waypoint_idx in TargetDataset

    def forward(self, x, action):
        x = self.normalize(x)
        x = self.conv(x)
        x = self.deconv(x)

        out = []
        branches = torch.unique(action.long())
        for b in branches:
            out.append(self.extract[b.item()](x[action==b.item()]))

        return torch.cat(out)
