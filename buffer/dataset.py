import torch
import numpy as np
from joblib import Memory
import zarr

from pathlib import Path
import time

#from utils import StaticWrap # TODO: refactor and pull from habitat2robomaster

from util import world_to_cam, fit_arc, make_onehot

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
    def __init__(self, episode_dir, target, scene):
        with open(episode_dir / 'episode.csv', 'r') as f:
            measurements = f.readlines()[1:]
        x = np.genfromtxt(measurements, delimiter=',', dtype=np.float32).reshape(-1, 10)
        # action,compass_r,compass_t,x,y,z,i,j,k,l
        self.actions = torch.LongTensor(x[:, 1])
        self.compass = torch.as_tensor(x[:, 1:3])
        self.positions = torch.as_tensor(x[:, 3:6])
        self.rotations = torch.as_tensor(x[:, 6:])

        # semantic can easily fit, ~1.5 GB compressed for 10k episodes
        #                            28 GB uncompressed
        self.target = zarr.open(str(episode_dir / target), mode='r')[:]

        self.waypoints = torch.zeros(len(self.actions), 5, 2)
        self.valid = torch.zeros_like(self.actions, dtype=torch.bool)
        onehot = make_onehot(np.uint8(self.target), scene='apartment_0')
        print(onehot.shape)
        for i in range(len(self.actions)-1):
            j = 20
            while True:
                #try:
                arc = fit_arc(self.actions, self.compass, i, j)
                #except:
                    #break
                if arc is None:
                    break
                x, y = arc
                u, v = world_to_cam(x, y)
                v = 160-torch.clamp(v, min=0, max=159)
                u = torch.clamp(u, min=0, max=383)
                
                print(onehot[i,int(v[-1]),int(u[-1]),0])
                if onehot[i,int(v[-1]),int(u[-1]),0]:
                    self.valid[i] = True
                    self.waypoints[i] = torch.stack([u, v], dim=-1)
                    break
                j -= 1
            print(i,v,u)

        self.num_valid = self.valid.sum()

    def __len__(self):
        return self.num_valid

    def __getitem__(self, idx):
        target = self.target[self.valid][idx]
        if self.target == 'semantic':
            target = make_onehot(target, scene='apartment_0')

        action = self.actions[self.valid][idx]

        r, t = self.compass[self.valid][idx]
        goal = torch.FloatTensor([r, np.cos(-t), np.sin(-t)])

        # TODO: waypoints!
        waypoints = self.waypoints[self.valid][idx]

        return target, action, goal, waypoints
