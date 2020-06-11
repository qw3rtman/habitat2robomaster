from buffer.dataset import HabitatDataset
from pathlib import Path
from buffer.util import make_onehot
from pyquaternion import Quaternion
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math

ACTIONS = ['S', 'F', 'L', 'R']
def rotate_origin_only(x, y, radians):
    """Only rotate a point around the origin (0, 0)."""
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy

cv2.namedWindow('semantic', cv2.WINDOW_NORMAL)
cv2.resizeWindow('semantic', 768, 320)

f = 384 / (2 * np.tan(120 * np.pi / 360))
A = torch.tensor([[ f, 0., 192.],
                  [0.,  f,   0.],
                  [0., 0.,   1.]])

zoom, k = 10, 8
def get_arc(xy, rotation):
    path = xy - xy[0]#[i:] - xy[i]
    path = np.stack(rotate_origin_only(*path.T, Quaternion(*rotation[1:4],
        rotation[0]).yaw_pitch_roll[1]), axis=-1)

    valid = np.any((path > zoom) | (path < -zoom), axis=1)
    first_invalid = np.searchsorted(np.cumsum(valid), 1)

    x,y = path[:first_invalid].T
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd**2+yd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])

    t = np.linspace(0,u.max(),k)
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)

    return xn, yn

def world_to_cam(x, y):
    M = torch.FloatTensor(np.stack([x, y, np.ones(y.shape[0])]))
    u, v = torch.mm(A, M)[:2]
    return u, v

d = HabitatDataset(Path('/Users/nimit/Documents/robomaster/habitat/005053'), 'semantic', 'apartment_0')

semantic = np.uint8(d.target_f[:])
x, y, z = d.positions.T
xy = np.stack([x, -z], axis=-1)

i = 0
while i < semantic.shape[0]:
    x, y = get_arc(xy[i:], d.rotations[i])
    u, v = world_to_cam(x, y)
    v = 160-torch.clamp(v, min=0, max=159)
    u = torch.clamp(u, min=0, max=383)

    s = cv2.cvtColor(255*np.uint8(make_onehot(semantic[i], scene='apartment_0')).reshape(160, -1), cv2.COLOR_GRAY2RGB)
    for l in range(u.shape[0]):
        cv2.circle(s, (int(u[l]), int(v[l])), 2, (255, 0, 0), -1)
    cv2.imshow('semantic', s)

    fig, ax = plt.subplots(ncols=1, nrows=1); ax.set_ylim(-zoom, zoom); ax.set_xlim(-zoom, zoom); ax.axis('off'); fig.tight_layout(pad=0); ax.margins(0)
    ax.scatter(x, y, c='r')
    plt.savefig('test.png')
    plt.close()
    cv2.imshow('bird', np.array(Image.open('test.png')))

    print(i, ACTIONS[d.actions[i]])

    key = cv2.waitKey(0)
    if key == 97:
        i -= 1
    elif key == 106:
        i -= 10
    elif key == 100:
        i += 1
    elif key == 108:
        i += 10
    elif key == 113:
        cv2.destroyAllWindows()
        break
