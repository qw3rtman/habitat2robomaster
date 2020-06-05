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

f = 384 / (2 * np.tan(120 * np.pi / 360))
A = torch.tensor([[ f, 0., 192.],
                  [0.,  f,   0.],
                  [0., 0.,   1.]])

def get_arc(actions, i, j):
    movement = actions[i:i+50] == 1
    k = np.searchsorted(np.cumsum(movement), j).item()

    R = r[i:i+k][movement[:k]]-r[i] # relative
    # R = abs(R / R.abs().mean())
    T = t[i:i+k][movement[:k]]-t[i+1] # absolute
    x, y = rotate_origin_only(R*np.cos(T), R*np.sin(T), np.pi/2)
    if T.shape[0] > 1 and (y < 0).sum() == 0: # if not behind camera
        _t = np.linspace(T[0], T[-1], 5)
        _r = np.linspace(R[0], R[-1], 5)
        _x, _y = rotate_origin_only(_r*np.cos(_t), _r*np.sin(_t), np.pi/2)
        return _x, _y

    return None

def world_to_cam(x, y):
    M = torch.FloatTensor(np.stack([x, y, np.ones(y.shape[0])]))
    u, v = torch.mm(A, M)[:2]
    return u, v

i = 0
init = False
while i < replay_buffer.size:
    fig, ax = plt.subplots(ncols=1, nrows=1); ax.set_ylim(-1,1); ax.set_xlim(-1,1); ax.axis('off'); fig.tight_layout(pad=0); ax.margins(0)

    semantic = np.load(Path(f'/Users/nimit/Documents/robomaster/habitat/160x384/semantic/{i+1:03}.npy'))==2
    j = 20
    while True:
        arc = get_arc(replay_buffer.actions, i, j) # look up to `i` frames ahead, get `j` movements
        if arc is None:
            break
        x, y = arc
        u, v = world_to_cam(x, y)
        v = 160-torch.clamp(v, min=0, max=159)
        u = torch.clamp(u, min=0, max=383)
        print(i,j,  v,u)

        if semantic[int(v[-1])][int(u[-1])]:
            break
        j -=1

    try:
        ax.plot(x, y, c='r')
        plt.savefig('test.png')
        plt.close()
        cv2.imshow('bird', np.array(Image.open('test.png')))

        rgb = replay_buffer.targets[i].squeeze().numpy()
        for l in range(u.shape[0]):
            cv2.circle(rgb, (int(u[l]), int(v[l])), 2, (255, 0, 0), -1)
        cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    except:
        pass

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
