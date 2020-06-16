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
    def get_episodes(split_dir, target_type, scene, zoom, steps):
        episode_dirs = list(split_dir.iterdir())

        data = []
        for i, episode_dir in enumerate(episode_dirs):
            data.append(SourceDataset(episode_dir, target_type, scene, zoom, steps))

            if i % 100 == 0:
                print(f'[{i:05}/{len(episode_dirs)}]')

        return data

    def make_dataset(is_train):
        split = 'train' if is_train else 'val'

        start = time.time()
        data = get_episodes(Path(dataset_dir) / split, target_type, scene, zoom, steps)
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

        self.xy = np.stack([self.positions[:, 0], -self.positions[:, 2]], axis=-1)
        self.waypoints = torch.zeros(self.actions.shape[0], 8, 2)
        self.ahead = torch.zeros(self.actions.shape[0], dtype=np.long)
        for i in range(self.actions.shape[0] - 1):
            xn, yn, self.ahead[i] = fit_arc(self.xy[i:], self.rotations[i],
                    zoom=self.zoom, steps=self.steps)
            self.waypoints[i] = torch.as_tensor(np.stack([xn, yn])).T
            if self.waypoints[i].abs().max() < 0.9 * self.zoom:
                break

        self.actions = self.actions[:i]
        self.compass = self.compass[:i]
        self.positions = self.positions[:i]
        self.rotation = self.rotations[:i]
        self.xy = self.xy[:i]
        self.waypoints = self.waypoints[:i]
        self.ahead = self.ahead[:i]

        self.target_f = None

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        if self.target_f is None:
            self.target_f = zarr.open(str(self.episode_dir / self.target_type), mode='r')

        target = self.target_f[idx]
        if self.target_type == 'semantic':
            target = make_onehot(np.uint8(target), scene=self.scene)

        #action = self.actions[idx]                                # local action taken
        #action = self.actions[idx:idx+self.ahead[idx]].mode()[0] # most common path action
        # max angle of waypoint wrt agent; [180º, 20º] is L, [20º, -20º] is F, [-20º, -180º]
        t = torch.atan2(-self.waypoints[idx,...,0], self.waypoints[idx,...,1])
        max_angle = t[t.abs().argmax()]
        if max_angle > 0.30:    # [20º, 180º],   L
            action = 2
        elif max_angle < -0.30: # [-20º, -180º], R
            action = 3
        else:                   # [-20º, 20º]    F
            action = 1

        r, t = self.compass[idx]
        goal = torch.FloatTensor([r, np.cos(-t), np.sin(-t)])

        waypoints = self.waypoints[idx].clone().detach() / self.zoom # [-1, 1] x [-1, 1]

        return target, action, goal, waypoints

if __name__ == '__main__':
    import argparse
    import cv2
    from PIL import Image
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--zoom', type=float, default=3)
    parsed = parser.parse_args()

    d = SourceDataset(parsed.dataset_dir, 'semantic', 'apartment_0', zoom=parsed.zoom, steps=8)

    cv2.namedWindow('semantic', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('semantic', 768, 320)

    i = 0
    while i < len(d):
        target, action, goal, waypoints = d[i]

        s = cv2.cvtColor((255*np.uint8(target)).reshape(160, -1), cv2.COLOR_GRAY2RGB)
        for x, y in waypoints * parsed.zoom:
            _x, _y = int(10*x)+192, 80-int(10*y)
            cv2.circle(s, (_x, _y), 3, (255, 0, 0), -1)
        cv2.imshow('semantic', s)

        print(i, ACTIONS[action])
        r = torch.sqrt(torch.pow(parsed.zoom*waypoints[...,0], 2) +
                torch.pow(parsed.zoom*waypoints[...,1], 2))
        t = torch.atan2(-waypoints[...,0], waypoints[...,1])
        print(d.compass[i])
        print(torch.stack([r, t], dim=-1)[-1])
        print()

        key = cv2.waitKey(0)
        if key == 97:
            i = max(0, i-1)
        elif key == 106:
            i = max(0, i-10)
        elif key == 100:
            i = min(len(d)-1, i+1)
        elif key == 108:
            i = min(len(d)-1, i+10)
        elif key == 113:
            cv2.destroyAllWindows()
            break
