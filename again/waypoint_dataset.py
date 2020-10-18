import torch
import numpy as np
from PIL import Image

from pathlib import Path
import time

from .const import GIBSON_NAME2IDX
from .util import fit_arc
from .pointgoal_dataset import Wrap

def get_dataset(dataset_dir, zoom=3, steps=5, dataset_size=1.0, batch_size=128, num_workers=0, **kwargs):

    def get_episodes(split_dir, zoom, steps):
        episode_dirs = list(split_dir.iterdir())
        num_episodes = int(max(100, dataset_size * len(episode_dirs))) # at least 100 episodes for train/val

        data = []
        for i, episode_dir in enumerate(episode_dirs[:num_episodes]):
            data.append(WaypointDataset(episode_dir.resolve(), zoom, steps))

            if i % 100 == 0:
                print(f'[{i:05}/{num_episodes}]')

        return data

    def make_dataset(is_train):
        split = 'train' if is_train else 'val'

        start = time.time()
        data = get_episodes(Path(dataset_dir) / split, zoom, steps)
        print(f'{split}: {len(data)} episodes in {time.time()-start:.2f}s')

        return Wrap(data, batch_size, 250 if is_train else 25, num_workers)

    return make_dataset(True), make_dataset(False)


class WaypointDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir, zoom=3, steps=8):
        self.episode_dir = episode_dir
        self.scene_idx = GIBSON_NAME2IDX[episode_dir.parents[1].stem.split('-')[1]] # dataset-scene
        self.zoom = zoom
        self.steps = steps

    def __len__(self):
        return int(self.episode_dir.stem.split('-')[1]) - (2 * self.steps)

    def __getitem__(self, idx):
        if not hasattr(self, 'waypoints'):
            with open(self.episode_dir / 'episode.csv', 'r') as f:
                x = np.genfromtxt(f.readlines()[1:], delimiter=',', dtype=np.float32).reshape(-1, 10)
                self.positions = torch.as_tensor(x[:, 3:6])
                self.rotations = torch.as_tensor(x[:, 6:])
                xy = np.stack([self.positions[:, 0], -self.positions[:, 2]], axis=-1)

                self.actions = torch.zeros(len(self), dtype=np.long)
                self.waypoints = torch.zeros(len(self), self.steps, 2)
                ahead = torch.zeros(len(self), dtype=np.long)

                for i in range(len(self)):
                    xn, yn, ahead[i] = fit_arc(xy[i:], self.rotations[i], zoom=self.zoom, steps=self.steps)
                    self.waypoints[i] = torch.as_tensor(np.stack([xn, yn])).T
                    # max angle of waypoint wrt agent; [180º, 20º] is L, [20º, -20º] is F, [-20º, -180º]
                    t = torch.atan2(-self.waypoints[idx,...,0], self.waypoints[idx,...,1])
                    max_angle = t[t.abs().argmax()]
                    if max_angle > 0.30:      # [20º, 180º],   L
                        self.actions[i] = 1
                    elif max_angle < -0.30:   # [-20º, -180º], R
                        self.actions[i] = 2
                    else:                     # [-20º, 20º]    F
                        self.actions[i] = 0

                    if self.waypoints[i].abs().max() < 0.9 * self.zoom:
                        break

            self.length = i

        idx = min(idx, self.length)
        target = np.array(Image.open(self.episode_dir/f'rgb_{idx:03}.png'))
        waypoints = self.waypoints[idx].clone().detach() / self.zoom # [-1, 1] x [-1, 1]
        return self.scene_idx, target, self.actions[idx], waypoints
