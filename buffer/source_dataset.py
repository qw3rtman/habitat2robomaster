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
def get_dataset(dataset_dir, target_type, scene, zoom, steps, batch_size=128, num_workers=0, **kwargs):

    @memory.cache
    def get_episodes(split_dir, target_type, dataset_size):
        episode_dirs = list(split_dir.iterdir())
        num_episodes = int(max(1, dataset_size * len(episode_dirs)))

        data = []
        for i, episode_dir in enumerate(episode_dirs[:num_episodes]):
            data.append(SourceDataset(episode_dir, target_type, scene, zoom, steps))

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


class SourceDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir, target_type, scene, zoom=3, steps=8):
        self.episode_dir = episode_dir
        self.target_type = target_type
        self.scene = scene
        self.zoom = zoom
        self.steps = steps

        with open(episode_dir / 'episode.csv', 'r') as f:
            measurements = f.readlines()[1:]
        x = np.genfromtxt(measurements, delimiter=',', dtype=np.float32).reshape(-1, 10)
        # action,compass_r,compass_t,x,y,z,i,j,k,l
        self.actions = torch.LongTensor(x[:, 0])
        self.compass = torch.as_tensor(x[:, 1:3])
        self.positions = torch.as_tensor(x[:, 3:6])
        self.rotations = torch.as_tensor(x[:, 6:])

        self.target_f = zarr.open(str(self.episode_dir / self.target_type), mode='r')

        self.xy = np.stack([self.positions[:, 0], -self.positions[:, 2]], axis=-1)
        self.waypoints = torch.zeros(self.actions.shape[0], 8, 2)
        for i in range(self.actions.shape[0] - 1):
            self.waypoints[i] = torch.as_tensor(np.stack(fit_arc(self.xy[i:],
                self.rotations[i], zoom=zoom, steps=steps))).T
            # TODO: when we first get inside the zoom, then it's not far enough

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        target = self.target_f[idx]
        if self.target_type == 'semantic':
            target = make_onehot(np.uint8(target), scene=self.scene)

        action = self.actions[idx]

        r, t = self.compass[idx]
        goal = torch.FloatTensor([r, np.cos(-t), np.sin(-t)])

        waypoints = self.waypoints[idx].clone().detach() # [-zoom, zoom] x [-zoom, zoom]
        waypoints /= zoom                                # [-1, 1]       x [-1, 1]

        return target, action, goal, waypoints

if __name__ == '__main__':
    import argparse
    import cv2
    from PIL import Image
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parsed = parser.parse_args()

    d = HabitatDataset(parsed.dataset_dir, 'semantic', 'apartment_0', zoom=3, steps=8)
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
