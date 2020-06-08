import torch
import numpy as np
from joblib import Memory
import zarr

from pathlib import Path
from itertools import repeat
import time

from .util import cam_to_world, make_onehot, Wrap, C, rotate_origin_only

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

        source_teacher.eval()
        goal_prediction.eval()

        onehot = make_onehot(np.uint8(zarr.open(str(self.episode_dir / 'semantic'), mode='r')[:]), scene=scene)
        onehot = torch.as_tensor(onehot.reshape(-1, C, 160, 384), dtype=torch.float).cuda()
        self.waypoints = torch.empty(self.rgb_f.shape[0], 4, 5, 2)
        with torch.no_grad():
            for a in range(4):
                self.waypoints[:, a] = goal_prediction(onehot,
                        a*torch.ones(self.rgb_f.shape[0]).cuda()).cpu()

        self.waypoints[..., 0] = (self.waypoints[..., 0] + 1) * 384 / 2
        self.waypoints[..., 1] = (self.waypoints[..., 1] + 1) * 160 / 2

        rcost, rsint = rotate_origin_only(*cam_to_world(
            self.waypoints[..., 0].flatten(),
            159-self.waypoints[..., 1].flatten()
        ), -np.pi/2)
        r = np.sqrt(np.square(rcost) + np.square(rsint)).reshape(-1, 4, 5)# negative?
        t = np.arctan(rsint/rcost).reshape(-1, 4, 5)#[..., -1]
        t[np.isnan(t)] = 0. # looking forward

        self.goal = torch.stack([r, np.cos(-t), np.sin(-t)], dim=-1).cuda()
        self.actions = torch.empty(self.goal.shape[:3], dtype=torch.long)
        onehot = onehot.reshape(-1, 160, 384, C)
        with torch.no_grad():
            for a in range(4):
                for t in range(5):
                    self.actions[:,a,t] = source_teacher((onehot, self.goal[:,a,t])).sample().squeeze()
                    #print(self.actions[:,a,t], self.goal[:,a,t,0])

    def __len__(self):
        return self.rgb_f.shape[0] * 5

    def __getitem__(self, idx):
        rgb = self.rgb_f[idx//5]
        action = self.actions[idx//5][idx%5]

        return rgb, goal
