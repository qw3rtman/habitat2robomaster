import numpy as np
import torch
import cv2
import pandas as pd
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pyquaternion import Quaternion
from pathlib import Path
from joblib import Memory

import argparse
from operator import itemgetter
from itertools import repeat
import subprocess
import time

from utils import StaticWrap, DynamicWrap

ACTIONS = torch.eye(4)

memory = Memory('/scratch/cluster/nimit/data/cache', verbose=1)
def get_dataset(dataset_dir, dagger=False, interpolate=False, rnn=False, capacity=2000, batch_size=128, num_workers=0, augmentation=False, rgb=True, semantic=True, **kwargs):
    """
        * shared memory can be a bottleneck on clusters, num_workers=0 is fast enough
    """

    @memory.cache
    def get_episodes(split_dir):
        episode_dirs = list(split_dir.iterdir())
        num_episodes = int(max(1, kwargs.get('dataset_size', 1.0) * len(episode_dirs)))

        data = []
        for i, episode_dir in enumerate(episode_dirs[:num_episodes]):
            data.append(HabitatDataset(episode_dir, is_seed=dagger, interpolate=interpolate, augmentation=augmentation, rgb=rgb, semantic=semantic))
            if i % 500 == 0:
                print(f'[{i:05}/{num_episodes}]')

        return data

    def make_dataset(is_train):
        split = 'train' if is_train else 'val'

        start = time.time()
        data = get_episodes(Path(dataset_dir) / split)
        print(time.time()-start)

        print('%s: %d' % (split, len(data)))

        if dagger:
            return DynamicWrap(data, batch_size, 100000 if is_train else 10000, num_workers, capacity=capacity)                    # samples == # steps
        elif rnn:
            return StaticWrap(EpisodeDataset(data), batch_size, 500 if is_train else 50, num_workers, collate_fn=collate_episodes) # samples == episodes
        else:
            return StaticWrap(data, batch_size, 25000 if is_train else 2500, num_workers)                                          # samples == # steps

    return make_dataset(True), make_dataset(False)


def collate_episodes(episodes):
    rgbs, segs, actions, prev_actions, metas = [], [], [], [], []
    for i, episode in enumerate(episodes):
        if not episode.init:
            episode._init()

        if episode.rgb:
            rgbs.append(torch.zeros((len(episode), 256, 256, 3), dtype=torch.uint8))
        if episode.semantic:
            segs.append(torch.zeros((len(episode), 256, 256, 2), dtype=torch.float32))

        actions.append(episode.actions)
        prev_actions.append(torch.zeros(len(episode), dtype=torch.long))
        prev_actions[i][1:] = episode.actions[:-1].clone()

        metas.append(torch.zeros((len(episode), 2), dtype=torch.float))

        for t, step in enumerate(episode):
            rgb, _, seg, action, meta, _, prev_action = step
            if type(rgb) != int:
                rgbs[i][t] = rgb
            if type(seg) != int:
                segs[i][t] = seg
            metas[i][t] = meta

    rgb_batch = 0
    if len(rgbs) > 0:
        rgb_batch = pad_sequence(rgbs)

    seg_batch = 0
    if len(segs) > 0:
        seg_batch = pad_sequence(segs)

    action_batch = pad_sequence(actions)
    prev_action_batch = pad_sequence(prev_actions)
    meta_batch = pad_sequence(metas)

    mask = (action_batch != 0).float()
    indices = torch.min(torch.argmax(mask, dim=0) + 1, torch.LongTensor([mask.shape[0] - 1]))
    for episode, index in enumerate(indices):
        mask[index.item(), episode] = 1.

    return rgb_batch, seg_batch, action_batch, prev_action_batch, meta_batch, mask


class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, episodes):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


class HabitatDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir, is_seed=False, interpolate=False, augmentation=False, rgb=True, semantic=True):
        if not isinstance(episode_dir, Path):
            episode_dir = Path(episode_dir)

        self.episode_dir = episode_dir
        self.rgb = rgb
        self.semantic = semantic

        self.interpolate = interpolate
        self.augmentation = augmentation

        self.init = False
        self._init()

        # DAgger
        self.episode_idx = 0
        self.loss = 0.0 # kick out these seeded ones first
        self.is_seed = is_seed

    def _init(self):
        if self.init:
            return

        self.imgs = []
        if self.rgb:
            self.imgs = [str(img) for img in sorted(self.episode_dir.glob('rgb_*.png'))]
            assert len(self.imgs) > 0

        self.segs = []
        if self.semantic:
            self.segs = [str(seg) for seg in sorted(self.episode_dir.glob('seg_*.npz'))]
            assert len(self.segs) > 0

        with open(self.episode_dir / 'episode.csv', 'r') as f:
            measurements = f.readlines()[1:]
        x = np.genfromtxt(measurements, delimiter=',', dtype=np.float32)
        self.positions = torch.as_tensor(x[:, 5:8])
        self.rotations = torch.as_tensor(x[:, 8:])
        self.actions = torch.LongTensor(x[:, 1])

        with open(self.episode_dir / 'info.csv', 'r') as f:
            info = f.readlines()[1].split(',')
        self.start_position = torch.Tensor(list(map(float, info[8:11])))
        self.start_rotation = torch.Tensor(list(map(float, info[11:15])))
        self.end_position   = torch.Tensor(list(map(float, info[15:18])))

        self.init = True

    def __len__(self):
        if not self.init:
            return 1 # doesn't matter in first phase
            #return int(subprocess.check_output(f'tail -n1 {str(self.episode_dir) + "/episode.csv"}', shell=True).split(b',')[0]) + 1
            #return int(subprocess.check_output(f'wc -l {str(self.episode_dir) + "/episode.csv"}', shell=True).split()[0]) - 1

        return self.actions.shape[0]

    def _get_direction(self, start, end=None):
        source_position = self.positions[start]
        source_rotation = Quaternion(*self.rotations[start,1:4], self.rotations[start,0])
        goal_position = self.end_position
        if end is not None:
            end = min(end, self.positions.shape[0]-1)
            goal_position = self.positions[end]

        return HabitatDataset.get_direction(source_position, source_rotation, goal_position)

    @staticmethod
    def get_direction(source_position, source_rotation, goal_position):
        direction_vector = goal_position - source_position
        direction_vector_agent = source_rotation.inverse.rotate(direction_vector)

        return torch.Tensor([-direction_vector_agent[2], -direction_vector_agent[0]])

    def aug(self, start, rgb, action):
        p = np.random.random()

        """
        if p < 0.10: # flip image + action
            #print('flip')
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            action = self._flip_action(action)
            return rgb, action, self._get_direction(start)
        """

        """
        if p < 0.15 and action == 1: # if we have a straight path, what if we had turned left?
            #print('rot')
            source_position = self.positions[start]
            source_rotation = Quaternion(*self.rotations[start,1:4], self.rotations[start,0])
            goal_position = self.end_position
            
            if np.random.random() < 0.50:
                rotation, action = self.left, torch.LongTensor([3])[0]
            else:
                rotation, action = self.right, torch.LongTensor([2])[0]
            direction = HabitatDataset.get_direction(source_position, rotation * source_rotation, goal_position)

            return rgb, action, direction
        """

        if p < 0.30: # goal is k steps ahead, instead of end_position
            #print('truncate')
            k = np.random.randint(3, 25)
            return rgb, action, self._get_direction(start, start+k)

        if p < 0.50: # stop if within 0.20
            x = np.random.random() * 0.20
            y = np.random.random() * np.sqrt(0.20**2 - x**2)

            return rgb, torch.LongTensor([0])[0], torch.Tensor([x, y])

        return rgb, action, self._get_direction(start)

    def _flip_action(self, action):
        # 0: stop
        # 1: forward
        # 2: left 10º
        # 3: right 10º
        if action == 2:
            return torch.LongTensor([3])[0]
        if action == 3:
            return torch.LongTensor([2])[0]

        return action

    @staticmethod
    def _make_semantic(semantic_observation):
        obstacles = np.uint8([1, 6, 8, 9, 10, 12, 19, 38, 39, 40])
        wall  = torch.Tensor(np.isin(semantic_observation, obstacles))

        walkable = np.uint8([2])
        floor = torch.Tensor(np.isin(semantic_observation, walkable))

        return torch.stack([wall, floor], dim=-1)

    def __getitem__(self, idx):
        if not self.init:
            self._init()

        rgb = 0
        if len(self.imgs) > idx and self.rgb:
            rgb    = torch.Tensor(cv2.imread(str(self.imgs[idx]), cv2.IMREAD_UNCHANGED))
            #rgb    = torch.Tensor(np.uint8(Image.open(self.imgs[idx])))

        seg = 0
        if len(self.segs) > idx and self.semantic:
            seg    = HabitatDataset._make_semantic(np.load(self.segs[idx])['semantic'])

        action = self.actions[idx]
        prev_action = self.actions[idx-1] if idx > 0 else torch.zeros_like(action)

        """
        if self.augmentation:
            rgb, action, meta = self.aug(idx, rgb, action)
        else:
        """
        meta = self._get_direction(idx)

        # rgb, mapview, segmentation, action, meta, episode, prev_action
        return rgb, 0, seg, action, meta, self.episode_idx, prev_action

    def __lt__(self, other):
        return self.loss < other.loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_dir', type=Path, required=True)
    args = parser.parse_args()

    transform_ = transforms.ToPILImage()

    dataset = HabitatDataset(args.episode_dir)
    for rgb, _, seg, action, _ in dataset:
        cv2.imshow('rgb', cv2.cvtColor(np.uint8(transform_(rgb)), cv2.COLOR_RGB2BGR))
        cv2.imshow('seg', np.float32(seg))
        cv2.waitKey(0)
