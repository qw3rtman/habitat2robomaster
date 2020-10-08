import torch
import numpy as np
from joblib import Memory
import zarr
import pandas as pd

from pathlib import Path
from itertools import repeat
import json
import time

def polar1(r, t):
    return torch.FloatTensor([r, np.cos(-t), np.sin(-t)]).T

def polar2(r, t):
    return torch.FloatTensor([r * np.cos(-t), r * np.sin(-t)]).T

def rff(r, t, D=100):
    w = np.random.normal(0, 1, (D, r.size))
    b = np.random.uniform(0, 2*np.pi, (D, r.size))

    return torch.FloatTensor([
        np.sqrt(2 / D) * r * np.cos((w*t) + b),
        np.sqrt(2 / D) * r * np.sin((w*t) + b),
    ]).reshape(r.size, -1)

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

#memory = Memory('/scratch/cluster/nimit/data/cache', mmap_mode='r+', verbose=0)
def get_dataset(dataset_dir, dataset_size=1.0, goal_fn='polar1', batch_size=128, num_workers=0, **kwargs):

    #@memory.cache
    def get_episodes(split_dir, goal_fn, dataset_size):
        episode_dirs = list(split_dir.iterdir())
        num_episodes = int(max(5, dataset_size * len(episode_dirs))) # at least 5 episodes for train/val

        data = []
        for i, episode_dir in enumerate(episode_dirs[:num_episodes]):
            data.append(HabitatDataset(episode_dir, globals()[goal_fn]))

            if i % 100 == 0:
                print(f'[{i:05}/{num_episodes}]')

        return data

    def make_dataset(is_train):
        split = 'train' if is_train else 'val'

        start = time.time()
        data = get_episodes(Path(dataset_dir) / split, goal_fn, dataset_size)
        print(f'{split}: {len(data)} episodes in {time.time()-start:.2f}s')

        # 1000, 100
        return Wrap(data, batch_size, 250 if is_train else 25, num_workers)

    return make_dataset(True), make_dataset(False)

root = Path('/scratch/cluster/nimit/habitat/habitat-api/data/scene_datasets/replica')
def make_onehot(semantic, scene=None):
    """
        input:  torch (B,H,W,1), dtype: torch/np.uint8
        output: torch (B,H,W,1), dtype: torch.float
    """
    semantic = semantic.reshape(-1, 160, 384)
    onehot = torch.zeros((*semantic.shape, 1), dtype=torch.float)

    with open(root / f'{scene}/habitat/info_semantic.json', 'r') as f:
        j = json.load(f)

    instance_to_class = np.array(j['id_to_label'])
    class_names = {_class['name']: _class['id'] for _class in j['classes']}
    classes = instance_to_class[semantic]

    if scene == 'apartment_0':
        floor = np.array([class_names['floor'], class_names['rug'],
            class_names['stair'], class_names['shower-stall'],
            class_names['basket']])
    elif scene in ['apartment_1', 'apartment_2']:
        floor = np.array([class_names['floor'], class_names['rug']])
    elif scene.split('_')[0] == 'frl': #'frl_apartment_4':
        floor = np.array([class_names['floor'], class_names['rug'],
            class_names['mat']])#, class_names['stair']])

    onehot[..., 0] = torch.as_tensor(np.isin(classes, floor), dtype=torch.float)
    onehot[:,:80,:,0] = 0 # floor is never above the horizon

    return onehot

class HabitatDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir, goal_fn):
        self.episode_dir = episode_dir
        self.scene = episode_dir.parents[1].stem.split('-')[1]

        with open(episode_dir / 'episode.csv', 'r') as f:
            measurements = f.readlines()[1:]
        x = np.genfromtxt(measurements, delimiter=',', dtype=np.float32).reshape(-1, 10)
    
        """ dagger episodes
        length = np.argmax(x[:,0]==0)
        self.actions = torch.LongTensor(x[:length, 0]) # cut before STOP
        """
        self.actions = torch.LongTensor(x[:-1, 0])
        self.xy = torch.FloatTensor(x[:-1,[3,5]])

        self.r, self.t = x[:-1, 1], x[:-1, 2]
        self.goal = goal_fn(self.r, self.t)
        #self.goal_fn = goal_fn

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        if not hasattr(self, 'target_f'):
            self.target_f = zarr.open(str(self.episode_dir / 'rgb'), mode='r')
        return self.target_f[idx], self.goal[idx], self.actions[idx]-1, self.xy[idx]
        #return make_onehot(self.semantic_f[idx], scene=self.scene)[0], self.goal[idx], self.actions[idx]-1
