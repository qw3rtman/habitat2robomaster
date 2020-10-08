import torch
import numpy as np
from joblib import Memory
import zarr
import pandas as pd

from .dataset import Wrap, make_onehot

from pathlib import Path
import time

ACTIONS = ['F', 'L', 'R']

#memory = Memory('/scratch/cluster/nimit/data/cache', mmap_mode='r+', verbose=0)
def get_dataset(dataset_dir, batch_size=128, num_workers=0, temporal_dim=1, **kwargs):

    #@memory.cache
    def get_episodes(split_dir, dataset_size):
        episode_dirs = list(split_dir.iterdir())
        num_episodes = int(max(1, dataset_size * len(episode_dirs)))

        data = []
        for i, episode_dir in enumerate(episode_dirs[:num_episodes]):
            data.append(PairDataset(episode_dir, temporal_dim=temporal_dim))

            if i % 100 == 0:
                print(f'[{i:05}/{num_episodes}]')

        return data

    def make_dataset(is_train):
        split = 'train' if is_train else 'val'

        start = time.time()
        data = get_episodes(Path(dataset_dir) / split, kwargs.get('dataset_size', 1.0))
        print(f'{split}: {len(data)} episodes in {time.time()-start:.2f}s')

        # 1000, 100
        return Wrap(data, batch_size, 25 if is_train else 5, num_workers)

    return make_dataset(True), make_dataset(False)

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir, temporal_dim=1):
        self.episode_dir = episode_dir
        self.scene = episode_dir.parents[1].stem.split('-')[1]

        self.t_min, self.t_max = 1, temporal_dim+1

        with open(episode_dir / 'episode.csv', 'r') as f:
            measurements = f.readlines()[1:]
        x = np.genfromtxt(measurements, delimiter=',', dtype=np.float32).reshape(-1, 10)
        self.actions = torch.LongTensor(x[:, 0])

    def __len__(self):
        return self.actions.shape[0]-1

    def __getitem__(self, idx):
        if not hasattr(self, 'input_f'):
            self.input_f = zarr.open(str(self.episode_dir / 'rgb'), mode='r')

        t = np.random.randint(self.t_min, self.t_max)
        if idx+t > len(self)-1:
            t = len(self)-idx

        """
        t1, t2 = make_onehot(np.stack([
            self.semantic_f[idx], self.semantic_f[idx+t]
        ]), scene=self.scene)
        """

        t1, t2 = self.input_f[idx], self.input_f[idx+t]
        return t1, t2, self.actions[idx]-1, t-1
