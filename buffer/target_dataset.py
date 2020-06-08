import torch
import numpy as np
from joblib import Memory
import zarr

from pathlib import Path
from itertools import repeat
import time

from .util import world_to_cam, fit_arc, make_onehot, Wrap, C

memory = Memory('/scratch/cluster/nimit/data/cache', mmap_mode='r+', verbose=0)
def get_dataset(source_teacher, goal_prediction, dataset_dir, scene, batch_size=128, num_workers=0, **kwargs):

    @memory.cache
    def get_episodes(split_dir, dataset_size):
        episode_dirs = list(split_dir.iterdir())
        num_episodes = int(max(1, dataset_size * len(episode_dirs)))

        data = []
        for i, episode_dir in enumerate(episode_dirs[:num_episodes]):
            data.append(TargetDataset(net, episode_dir, scene))

            if i % 100 == 0:
                print(f'[{i:05}/{num_episodes}]')

        return data

    def make_dataset(is_train):
        split = 'train' if is_train else 'val'

        start = time.time()
        data = get_episodes(Path(dataset_dir) / split, kwargs.get('dataset_size', 1.0))
        print(f'{split}: {len(data)} episodes in {time.time()-start:.2f}s')

        return Wrap(data, batch_size, 1000 if is_train else 100, num_workers)

    return make_dataset(True), make_dataset(False)

class TargetDataset(torch.utils.data.Dataset):
    def __init__(self, source_teacher, goal_prediction, episode_dir, scene):
        self.episode_dir = episode_dir
        self.scene = scene

        self.rgb_f = zarr.open(str(self.episode_dir / 'rgb'), mode='r')

        onehot = make_onehot(zarr.open(str(self.episode_dir / 'semantic'), mode='r')[:])
        onehot = torch.as_tensor(onehot.reshape(-1, C, 160, 384), dtype=torch.float).cuda()
        self.waypoints = goal_prediction(onehot)
        # TODO: unnormalize, unproject to recover r, t

        actions = torch.empty(self.waypoints.shape[:2], dtype=torch.long)
        for t in range(5):
            actions[:, t] = source_teacher(onehot, goal[:, t])

    def __len__(self):
        return self.rgb_f.shape[0] * 5

    def __getitem__(self, idx):
        rgb = self.rgb_f[idx//5]
        action = self.actions[idx//5][idx%5]

        return rgb, goal
