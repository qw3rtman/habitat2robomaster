import torch
import numpy as np
from joblib import Memory
import zarr

from pathlib import Path
from itertools import repeat
import time

from .util import world_to_cam, fit_arc, make_onehot, Wrap

ACTIONS = ['S', 'F', 'L', 'R']

memory = Memory('/scratch/cluster/nimit/data/cache', mmap_mode='r+', verbose=0)
def get_dataset(dataset_dir, target_type, scene, batch_size=128, num_workers=0, **kwargs):

    @memory.cache
    def get_episodes(split_dir, target_type, dataset_size):
        episode_dirs = list(split_dir.iterdir())
        num_episodes = int(max(1, dataset_size * len(episode_dirs)))

        data = []
        for i, episode_dir in enumerate(episode_dirs[:num_episodes]):
            data.append(HabitatDataset(episode_dir, target_type, scene))

            if i % 100 == 0:
                print(f'[{i:05}/{num_episodes}]')

        return data

    def make_dataset(is_train):
        split = 'train' if is_train else 'val'

        start = time.time()
        data = get_episodes(Path(dataset_dir) / split, target_type, kwargs.get('dataset_size', 1.0))
        print(f'{split}: {len(data)} episodes in {time.time()-start:.2f}s')

        return Wrap(data, batch_size, 1000 if is_train else 100, num_workers)

    return make_dataset(True), make_dataset(False)


class HabitatDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir, target_type, scene):
        self.episode_dir = episode_dir
        self.target_type = target_type
        self.scene = scene

        with open(episode_dir / 'episode.csv', 'r') as f:
            measurements = f.readlines()[1:]
        x = np.genfromtxt(measurements, delimiter=',', dtype=np.float32).reshape(-1, 10)
        # action,compass_r,compass_t,x,y,z,i,j,k,l
        self.actions = torch.LongTensor(x[:, 0])
        self.compass = torch.as_tensor(x[:, 1:3])
        self.positions = torch.as_tensor(x[:, 3:6])
        self.rotations = torch.as_tensor(x[:, 6:])

        left  = torch.cat([torch.zeros(1, dtype=torch.bool), self.actions[:-1] == 2, torch.zeros(1, dtype=torch.bool)])
        right = torch.cat([torch.zeros(1, dtype=torch.bool), self.actions[:-1] == 3, torch.zeros(1, dtype=torch.bool)])

        target_f = zarr.open(str(self.episode_dir / self.target_type), mode='r')
        self.waypoints = torch.zeros(len(self.actions), 5, 2)
        self.valid = torch.zeros_like(self.actions, dtype=torch.bool)
        onehot = make_onehot(np.uint8(target_f[:]), scene=self.scene)
        for i in range(len(self.actions)-1):
            arc = fit_arc(self.actions, self.compass, onehot, i)
            if arc is None:
                continue
            self.valid[i] = True
            self.waypoints[i] = torch.stack(arc, dim=-1)

        self.num_valid = self.valid.sum()
        self.target_f = zarr.open(str(self.episode_dir / self.target_type), mode='r')

    def __len__(self):
        return self.num_valid

    def __getitem__(self, idx):
        idx = torch.arange(self.target_f.shape[0])[self.valid][idx].item()

        target = self.target_f[idx]
        if self.target_type == 'semantic':
            target = make_onehot(np.uint8(target), scene='apartment_0')

        action = self.actions[idx]

        r, t = self.compass[idx]
        goal = torch.FloatTensor([r, np.cos(-t), np.sin(-t)])

        waypoints = self.waypoints[idx].clone().detach()
        waypoints[:,0] = (2*waypoints[:,0]/384) - 1
        waypoints[:,1] = (2*waypoints[:,1]/160) - 1

        return target, action, goal, waypoints

if __name__ == '__main__':
    import argparse
    import cv2
    from PIL import Image
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parsed = parser.parse_args()

    d = HabitatDataset(parsed.dataset_dir, 'semantic', 'apartment_0')
    i = 0
    while i < len(d):
        target, action, goal, waypoints = d[i]
        semantic = cv2.cvtColor(255*np.uint8(target).reshape(160,384), cv2.COLOR_GRAY2RGB)
        for l in range(waypoints.shape[0]):
            cv2.circle(semantic, (int(waypoints[l,0]), int(waypoints[l,1])), 2, (255, 0, 0), -1)
        cv2.imshow('semantic', cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB))
        #print(ACTIONS[d.goal_actions[i]])

        key = cv2.waitKey(0)
        if key == 97:
            i -= 1
        elif key == 106:
            i -= 10
        elif key == 100:
            i += 1
        elif key == 108:
            i += 10
        elif key == 113:
            cv2.destroyAllWindows()
            break
