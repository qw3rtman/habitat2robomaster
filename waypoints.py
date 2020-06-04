import zarr
from buffer.frame_buffer import ReplayBuffer
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import cv2
import math

def rotate_origin_only(x, y, radians):
    """Only rotate a point around the origin (0, 0)."""
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy

replay_buffer = ReplayBuffer(int(2e3), history_size=1, dshape=(160, 384, 3), dtype=torch.uint8, goal_size=3)
replay_buffer.load(Path('/Users/nimit/Documents/robomaster/habitat/160x384'))

r, c, s = replay_buffer.goals.T
t = np.arccos(c)

cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
cv2.resizeWindow('rgb', 768, 320)

i = 0
init = False
while i < replay_buffer.size:
    rgb = replay_buffer.targets[i].squeeze().numpy()
    cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

    movement = replay_buffer.actions[i:i+10] == 1
    k = np.searchsorted(np.cumsum(movement), 6)

    # stupid
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.set_ylim(-1,1)
    ax.set_xlim(-1,1)

    R = r[i:i+k][movement[:k]]-r[i] # relative
    T = t[i:i+k][movement[:k]]-t[i] # absolute
    x, y = rotate_origin_only(R*np.cos(T), R*np.sin(T), np.pi/2)
    if (y < 0).sum() == 0: # if behind camera, then use prev
        if T.shape[0] != 0:
            _t = np.linspace(T[0], T[-1], 10)
            _r = np.linspace(R[0], R[-1], 10)
            _x, _y = rotate_origin_only(_r*np.cos(_t), _r*np.sin(_t), np.pi/2) # fit arc
            init = True
    if init:
        ax.scatter(x, y)
        ax.plot(_x, _y, c='r')
    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)
    plt.savefig('test.png')
    plt.close()
    cv2.imshow('bird', np.array(Image.open('test.png')))

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
