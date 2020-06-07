import torch
from pathlib import Path
import numpy as np
import json
import math

C = 1
def make_onehot(semantic, scene=None):
    """
        input:  torch (B,H,W,1), dtype: torch/np.uint8
        output: torch (B,H,W,C), dtype: torch.float
    """
    onehot = torch.zeros((*semantic.shape, C), dtype=torch.float)
    if scene is not None: # replica mapping
        mapping_f = Path(f'/scratch/cluster/nimit/habitat/habitat-api/data/scene_datasets/replica/{scene}/habitat/info_semantic.json')
        with open(mapping_f) as f:
            j = json.load(f)
        instance_to_class = np.array(j['id_to_label'])
        class_names = {_class['name']: _class['id'] for _class in j['classes']}
        classes = instance_to_class[semantic]
        floor = np.array([class_names['floor'], class_names['rug'], class_names['stair'],
                 class_names['shower-stall'], class_names['basket']])
        onehot[..., 0] = torch.as_tensor(np.isin(classes, floor), dtype=torch.float)
        #onehot[..., 1] = torch.as_tensor(classes == class_names['wall'], dtype=torch.float)
    else: # handle in env.py#step/#reset; TODO: move that logic to here
        onehot[..., 0] = torch.as_tensor(semantic==2, dtype=torch.float)
        #onehot[..., 1] = torch.as_tensor((semantic!=2)&(semantic!=17)&(semantic!=28), dtype=torch.float)
    return onehot


f = 384 / (2 * np.tan(120 * np.pi / 360))
A = torch.tensor([[ f, 0., 192.],
                  [0.,  f,   0.],
                  [0., 0.,   1.]])

def world_to_cam(x, y):
    M = torch.FloatTensor(np.stack([x, y, np.ones(y.shape[0])]))
    u, v = torch.mm(A, M)[:2]
    return u, v

def rotate_origin_only(x, y, radians):
    """Only rotate a point around the origin (0, 0)."""
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy

def fit_arc(actions, compass, i, j):
    movement = actions[i:i+50] == 1
    k = np.searchsorted(np.cumsum(movement), j).item()

    r, t = compass
    R = r[i:i+k][movement[:k]]-r[i]   # relative
    T = t[i:i+k][movement[:k]]-t[i+1] # relative
    x, y = rotate_origin_only(R*np.cos(T), R*np.sin(T), np.pi/2)
    if T.shape[0] > 1 and (y < 0).sum() == 0: # if not behind camera
        _t = np.linspace(T[0], T[-1], 5)
        _r = np.linspace(R[0], R[-1], 5)
        _x, _y = rotate_origin_only(_r*np.cos(_t), _r*np.sin(_t), np.pi/2)
        return _x, _y

    return None
