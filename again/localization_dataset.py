import torch
import numpy as np
from PIL import Image

from pathlib import Path
import time

from .const import GIBSON_NAME2IDX
from .pointgoal_dataset import Wrap

def get_dataset(dataset_dir, dataset_size=1.0, batch_size=128, num_workers=0, **kwargs):

    def get_episodes(split_dir, dataset_size):
        episode_dirs = list(split_dir.iterdir())
        num_episodes = int(max(100, dataset_size * len(episode_dirs))) # at least 100 episodes for train/val

        data = []
        for i, episode_dir in enumerate(episode_dirs[:num_episodes]):
            data.append(LocalizationDataset(episode_dir.resolve()))

            if i % 100 == 0:
                print(f'[{i:05}/{num_episodes}]')

        return data

    def make_dataset(is_train):
        split = 'train' if is_train else 'val'

        start = time.time()
        data = get_episodes(Path(dataset_dir) / split, dataset_size)
        print(f'{split}: {len(data)} episodes in {time.time()-start:.2f}s')

        return Wrap(data, batch_size, 250 if is_train else 25, num_workers)

    return make_dataset(True), make_dataset(False)

class LocalizationDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir):
        self.episode_dir = episode_dir
        self.scene_idx = GIBSON_NAME2IDX[episode_dir.parents[1].stem.split('-')[1]] # dataset-scene

        """ dagger episodes
        length = np.argmax(x[:,0]==0)
        self.actions = torch.LongTensor(x[:length, 0]) # cut before STOP
        """

    def __len__(self):
        # TODO: remove STOP when training policy, but
        #       keep for aux tasks, ~100k extra samples
        return int(self.episode_dir.stem.split('-')[1])

    def __getitem__(self, idx):
        if not hasattr(self, 'localization'):
            with open(self.episode_dir / 'episode.csv', 'r') as f:
                x = np.genfromtxt(f.readlines()[1:], delimiter=',', dtype=np.float32).reshape(-1, 10)
            xy = x[:,[3,5]]
            orientation = np.arcsin(
                -2*((x[:,7]*x[:,9])-(x[:,6]*x[:,8]))
            ).reshape(-1, 1)

            self.localization = torch.FloatTensor(np.concatenate(
                [xy, np.sin(orientation), np.cos(orientation)], axis=1
            ))

        target = np.array(Image.open(self.episode_dir/f'rgb_{idx:03}.png'))
        return self.scene_idx, target, self.localization[idx]
