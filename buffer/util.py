import torch
from pyquaternion import Quaternion
import numpy as np
import cv2

from pathlib import Path
from itertools import repeat
import json
import math
import yaml

from habitat.utils.visualizations import maps

def get_model_args(model, key=None):
    config = yaml.load((model.parent / 'config.yaml').read_text())
    if not key:
        return config

    return config[key]['value']

C = 1
root = Path('/scratch/cluster/nimit/habitat/habitat-api/data/scene_datasets/replica')
floor_y = {'frl_apartment_4': -1.3318579196929932}
def make_onehot(semantic, scene=None):
    """
        input:  torch (B,H,W,1), dtype: torch/np.uint8
        output: torch (B,H,W,C), dtype: torch.float
    """
    semantic = semantic.reshape(-1, 160, 384)
    onehot = torch.zeros((*semantic.shape, C), dtype=torch.float)
    if scene is not None: # replica mapping
        with open(root / f'{scene}/habitat/info_semantic.json', 'r') as f:
            j = json.load(f)

        instance_to_class = np.array(j['id_to_label'])
        class_names = {_class['name']: _class['id'] for _class in j['classes']}
        classes = instance_to_class[semantic]

        if scene == 'apartment_0':
            floor = np.array([class_names['floor'], class_names['rug'],
                class_names['stair'], class_names['shower-stall'],
                class_names['basket']])
        elif scene == 'apartment_2':
            floor = np.array([class_names['floor'], class_names['rug']])
        elif scene == 'frl_apartment_4':
            floor = np.array([class_names['floor'], class_names['rug'],
                class_names['mat'], class_names['stair']])

        onehot[..., 0] = torch.as_tensor(np.isin(classes, floor), dtype=torch.float)
    else: # handle in env.py#step/#reset; TODO: move that logic to here
        onehot[..., 0] = torch.as_tensor(semantic==2, dtype=torch.float)
        #onehot[..., 1] = torch.as_tensor((semantic!=2)&(semantic!=17)&(semantic!=28), dtype=torch.float)

    onehot[:,:80,:,0] = 0 # floor is never above the horizon
    return onehot


f = 384 / (2*np.tan(np.deg2rad(120/2)))
# py = 0.25 actually, but 0.40 approximately accounts for offsets
py, cx, cy = 0.40, 192, 80
def get_navigable(ep, scene):
    """
        Given an episode.csv + scene, project the walkable region into camera
        coordinates at y=0. Similar to floor semantic segmentation, but does
        not make any assumptions about occlusions. Can take the AND with floor
        segmentation to obtain the true navigable region in camera coordinates.
    """
    fpv = np.zeros((len(ep), 160, 384), dtype=np.bool)

    points = np.load(root / f'{scene}/floorplan.npy')
    points = points[np.isclose(points[:,1], floor_y[scene])]
    for i in range(len(ep)):
        centered = points-np.array(ep.iloc[i][['x','y','z']])
        centered[:, [2,0]] = np.stack(rotate_origin_only(centered[:,2],
            centered[:,0], Quaternion(*np.array(ep.iloc[i][['i', 'j', 'k', 'l']]
            )).angle), axis=-1)

        wty = (cy - (f/(-(centered[:,2])))*py)
        wtx = cx + ((centered[:,0])*(cy-wty))/py

        uv = np.int64(np.stack([wtx, 160-wty]))
        uv = uv[:,(uv[0]>=0)&(uv[0]<384)&(uv[1]>=0)&(uv[1]<160)&(uv[1]>=80)]
        fpv[i, uv[1], uv[0]] = 1
        fpv[i] = cv2.dilate(np.uint8(fpv[i]), np.ones((2,2), np.uint8), iterations=3).astype(np.bool) # (2,2)x3 is good for 1000000 samples in frl_apartment_4

    return fpv

fx = 1 / (np.tan((120*np.pi/180)/2)) # 384
fy = 1 / (np.tan((120*np.pi/180)/2)) # 160
A = torch.tensor([[fx,  0., 0.],
                  [0.,  fy, 0.],
                  [0.,  0.,   1.]])

def world_to_cam(x, y):
    M = torch.FloatTensor(np.stack([x, np.ones(y.shape[0]), y]))
    return torch.mm(A, M)[[0, 2]]

def cam_to_world(u, v):
    U = torch.stack([u, torch.ones_like(u), v])
    return torch.mm(A.inverse(), U)[[0, 2]]

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

def repeater(loader):
    for loader in repeat(loader):
        for data in loader:
            yield data

class Wrap(object):
    def __init__(self, data, batch_size, samples, num_workers, loss_sampler=False):
        datasets = torch.utils.data.ConcatDataset(data)

        if loss_sampler:
            sampler = LossSampler(datasets, batch_size)
            self.dataloader = torch.utils.data.DataLoader(datasets,
                batch_sampler=sampler, num_workers=num_workers, pin_memory=True)
        else:
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

def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(info["top_down_map"]["map"])
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map
