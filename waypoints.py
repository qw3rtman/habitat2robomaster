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
while i < replay_buffer.size:
    rgb = replay_buffer.targets[i].squeeze().numpy()
    cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

    # stupid
    fig, ax = plt.subplots(ncols=1, nrows=1)
    R = r[i:i+5]-r[i] # relative
    T = t[i:i+5]-t[i] # absolute
    ax.set_ylim(-1,1)
    ax.set_xlim(-1,1)
    x, y = R*np.cos(T), R*np.sin(T)
    x, y = rotate_origin_only(x, y, np.pi/2)
    ax.scatter(x, y)
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
