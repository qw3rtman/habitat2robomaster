import math

import torch
import numpy as np

from pyquaternion import Quaternion

class CartesianToPolar(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xy):
        """
        Assumes (x, y) are already scaled; i.e: [-zoom, zoom] x [-zoom, zoom]
        t in [-pi, pi] -> [0, 2pi]
        """
        r = torch.sqrt(torch.pow(xy[...,0], 2) + torch.pow(xy[...,1], 2))
        t = -torch.atan2(-xy[...,0], xy[...,1]) + np.pi

        return torch.stack([r, t], dim=-1)


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
        weights = torch.nn.functional.softmax(flat / self.temperature, dim=-1).view_as(logit)

        x = (weights.sum(-2) * torch.linspace(-1, 1, logit.shape[-1]).to(logit.device)).sum(-1)
        y = (weights.sum(-1) * torch.linspace(-1, 1, logit.shape[-2]).to(logit.device)).sum(-1)

        return torch.stack((x, y), -1)


def rotate_origin_only(x, y, radians):
    """Only rotate a point around the origin (0, 0)."""
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy


def fit_arc(xy, rotation, zoom=3, steps=8):
    path = xy - xy[0]
    path = np.stack(rotate_origin_only(*path.T, Quaternion(*rotation[1:4],
        rotation[0]).yaw_pitch_roll[1]), axis=-1)

    valid = np.any((path > zoom) | (path < -zoom), axis=1)
    first_invalid = np.searchsorted(np.cumsum(valid), 1).item()

    x,y = path[:first_invalid].T
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd**2+yd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])

    t = np.linspace(0,u.max(),steps)
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)

    return xn, yn, first_invalid
