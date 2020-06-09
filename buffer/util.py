import torch
from pathlib import Path
from itertools import repeat
import numpy as np
import json
import math
import yaml

def get_model_args(model, key=None):
    config = yaml.load((model.parent / 'config.yaml').read_text())
    if not key:
        return config

    return config[key]['value']

C = 1
def make_onehot(semantic, scene=None):
    """
        input:  torch (B,H,W,1), dtype: torch/np.uint8
        output: torch (B,H,W,C), dtype: torch.float
    """
    onehot = torch.zeros((*semantic.shape, C), dtype=torch.float)
    if scene is not None: # replica mapping
        mapping_f = Path(f'/scratch/cluster/nimit/habitat/habitat-api/data/scene_datasets/replica/{scene}/habitat/info_semantic.json')
        if not mapping_f.exists():
            mapping_f = Path(f'/Users/nimit/Documents/robomaster/habitat/habitat2robomaster/{scene}.json')
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

def cam_to_world(u, v):
    return torch.mm(A.inverse(), torch.stack([u, v, torch.ones_like(u)]))[:2]

def rotate_origin_only(x, y, radians):
    """Only rotate a point around the origin (0, 0)."""
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy

def fit_arc(actions, compass, onehot, i):
    movement = actions[i:] == 1

    for j in range(10, 1, -1):
        k = np.searchsorted(np.cumsum(movement), j).item()
        #l = np.where(~movement[k:] == 1)[0]
        #if len(l) > 0:
            #k = min(actions[i:].shape[0], k+l[0])

        r, t = compass.T
        t = np.arccos(np.cos(-t))
        R = r[i:i+k][movement[:k]]-r[i]   # relative
        T = t[i:i+k][movement[:k]]-t[i] # absolute
        x, y = rotate_origin_only(R*np.cos(T), R*np.sin(T), np.pi/2)
        if T.shape[0] > 1 and (y < 0).sum() == 0: # if behind camera, then use prev
            _t = np.linspace(T[0], T[-1], 5)
            _r = np.linspace(R[0], R[-1], 5)
            _x, _y = rotate_origin_only(_r*np.cos(_t), _r*np.sin(_t), np.pi/2)
            u, v = world_to_cam(_x, _y)
            v = 159-torch.clamp(v, min=0, max=159)
            u = torch.clamp(u, min=0, max=383)

            # walkable should be below the horizon (no stairs)
            if int(v[-1]) >= (159/2) and onehot[i,int(v[-1]),int(u[-1]),0]:
                return u, v

    return None

def repeater(loader):
    for loader in repeat(loader):
        for data in loader:
            yield data

class Wrap(object):
    def __init__(self, data, batch_size, samples, num_workers):
        datasets = torch.utils.data.ConcatDataset(data)
        self.dataloader = torch.utils.data.DataLoader(datasets, shuffle=True,
                batch_size=batch_size, num_workers=num_workers, drop_last=True,
                pin_memory=True)
        self.data = repeater(self.dataloader)
        self.samples = samples

    def __iter__(self):
        for _ in range(self.samples):
            yield next(self.data)

    def __len__(self):
        return self.samples
