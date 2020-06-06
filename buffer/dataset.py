import torch
import numpy as np
from joblib import Memory
import zarr

from pathlib import Path
import time

from utils import StaticWrap # TODO: refactor and pull from habitat2robomaster

memory = Memory('/scratch/cluster/nimit/data/cache', mmap_mode='r+', verbose=0)
def get_dataset(dataset_dir, target, batch_size=128, num_workers=0, **kwargs):

    @memory.cache
    def get_episodes(split_dir, target, dataset_size):
        episode_dirs = list(split_dir.iterdir())
        num_episodes = int(max(1, dataset_size * len(episode_dirs)))

        data = []
        for i, episode_dir in enumerate(episode_dirs[:num_episodes]):
            data.append(HabitatDataset(episode_dir, target))

            if i % 500 == 0:
                print(f'[{i:05}/{num_episodes}]')

        return data

    def make_dataset(is_train):
        split = 'train' if is_train else 'val'

        start = time.time()
        data = get_episodes(Path(dataset_dir) / split, target, kwargs.get('dataset_size', 1.0))
        print(f'{split}: {len(data)} in {time.time()-start:.2f}s')

        return StaticWrap(data, batch_size, 25000 if is_train else 2500, num_workers)

    return make_dataset(True), make_dataset(False)


class HabitatDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir, target):
        with open(self.episode_dir / 'episode.csv', 'r') as f:
            measurements = f.readlines()[1:]
        x = np.genfromtxt(measurements, delimiter=',', dtype=np.float32).reshape(1, -1)
        self.positions = torch.as_tensor(x[:, 5:8])
        self.rotations = torch.as_tensor(x[:, 8:])
        self.actions = torch.LongTensor(x[:, 1])
        self.compass = torch.as_tensor(x[:, 3:5])

        # semantic can easily fit, ~1.5 GB compressed for 10k episodes
        #                            28 GB uncompressed
        self.target = zarr.open(str(episode_dir / target), mode='r')[:]

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        target = self.target[idx]
        if self.target == 'semantic':
            target = make_onehot(target, scene='apartment_0')

        action = self.actions[idx]

        r, t = self.compass[idx]
        goal = torch.FloatTensor([r, np.cos(-t), np.sin(-t)])

        # TODO: waypoints!

        # target, action, goal
        return target, action, goal
