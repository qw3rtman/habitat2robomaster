import numpy as np
import torch
import cv2
import pandas as pd
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from pyquaternion import Quaternion
from pathlib import Path
from joblib import Memory
import zarr

import argparse
from operator import itemgetter
from itertools import repeat
from collections import OrderedDict
import subprocess
import random
import time

from utils import StaticWrap, DynamicWrap

ACTIONS = torch.eye(4)

memory = Memory('/scratch/cluster/nimit/data/cache', mmap_mode='r+', verbose=0)
def get_dataset(dataset_dir, dagger=False, interpolate=False, rnn=False, capacity=2000, batch_size=128, num_workers=0, augmentation=False, rgb=True, semantic=True, **kwargs):

    @memory.cache
    def get_episodes(split_dir):
        episode_dirs = list(split_dir.iterdir())
        num_episodes = int(max(1, kwargs.get('dataset_size', 1.0) * len(episode_dirs)))

        data = []
        for i, episode_dir in enumerate(episode_dirs[:num_episodes]):
            episode = HabitatDataset(episode_dir, is_seed=dagger, interpolate=interpolate, augmentation=augmentation, rgb=rgb, semantic=semantic)
            # bounds memory usage by (250 x B x 256 x 256 x 3) x 4 bytes
            # i.e: 0.20 GB per batch dimension
            if len(episode) <= 250:
                data.append(episode)

            if i % 500 == 0:
                print(f'[{i:05}/{num_episodes}]')

        return data

    def make_dataset(is_train):
        split = 'train' if is_train else 'val'

        start = time.time()
        data = get_episodes(Path(dataset_dir) / split)
        print(f'{split}: {len(data)} in {time.time()-start}s')

        if dagger:
            return DynamicWrap(data, batch_size, 100000 if is_train else 10000, num_workers, capacity=capacity)                    # samples == # steps
        elif rnn:
            bucket_batch_sampler = BucketBatchSampler(data, batch_size)
            return StaticWrap(EpisodeDataset(data), batch_size, 500 if is_train else 50, num_workers, collate_fn=collate_episodes, batch_sampler=bucket_batch_sampler) # samples == episodes
        else:
            return StaticWrap(data, batch_size, 25000 if is_train else 2500, num_workers)                                          # samples == # steps

    return make_dataset(True), make_dataset(False)


class BucketBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, episodes, batch_size):
        self.batch_size = batch_size

        buckets = OrderedDict() # len -> [Episode, Episode, ...]
        for idx, episode in enumerate(episodes):
            length = len(episode)
            if length not in buckets:
                buckets[length] = []
            buckets[length].append(idx)

        idx, self.batches = 0, [0] * len(episodes)
        for indices in buckets.values():
            random.shuffle(indices)
            self.batches[idx:idx+len(indices)] = indices
            idx += len(indices)

    def __len__(self):
        return len(self.batches) // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            start_idx = int((len(self.batches) - self.batch_size) * random.random())
            batch = self.batches[start_idx:start_idx+self.batch_size]
            random.shuffle(batch)

            yield batch

def collate_episodes(episodes):
    rgbs, segs, actions, prev_actions, metas = [], [], [], [], []
    for i, episode in enumerate(episodes):
        actions.append(episode.actions)

        _prev_actions = torch.empty_like(episode.actions)
        _prev_actions[0] = 0
        _prev_actions[1:].copy_(episode.actions[:-1])
        prev_actions.append(_prev_actions)

        metas.append(episode.meta)

        start = time.time()
        worker_info = torch.utils.data.get_worker_info()
        if episode.rgb:
            rgbs.append(torch.as_tensor(episode.get_rgb_sequence()).float())

        if episode.semantic:
            segs.append(seg_seq = torch.as_tensor(episode.get_semantic_sequence()).float())
        #print(f'[{worker_info.id}] [{i:03}/{len(episodes)}] {(time.time()-start):.02f}')

    action_batch = pad_sequence(actions)
    prev_action_batch = pad_sequence(prev_actions)
    meta_batch = pad_sequence(metas)

    mask = (action_batch != 0).float()
    indices = torch.min(torch.argmax(mask, dim=0) + 1, torch.LongTensor([mask.shape[0] - 1]))
    for episode, index in enumerate(indices):
        mask[index.item(), episode] = 1.

    #lengths = torch.Tensor([len(episode) for episode in episodes])
    #pack_padded_sequence(rgb_batch, lengths, enforce_sorted=False)

    return rgbs, segs, action_batch, prev_action_batch, meta_batch, mask


class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, episodes):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


obstacles = np.uint8([1, 6, 8, 9, 10, 12, 19, 38, 39, 40])
walkable = np.uint8([2])
class HabitatDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir, is_seed=False, interpolate=False, augmentation=False, rgb=True, semantic=True):
        if not isinstance(episode_dir, Path):
            episode_dir = Path(episode_dir)

        self.episode_dir = episode_dir
        self.rgb = rgb
        self.semantic = semantic

        self.interpolate = interpolate
        self.augmentation = augmentation

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

        self.meta = torch.empty((self.actions.shape[0], 2))
        for i in range(self.actions.shape[0]):
            self.meta[i] = self._get_direction(i)

        if self.rgb:
            assert (self.episode_dir / 'rgb').exists()
            self.rgb_f = zarr.open(str(self.episode_dir / 'rgb'), mode='r')

        if self.semantic:
            assert (self.episode_dir / 'semantic').exists()

        # DAgger
        self.episode_idx = 0
        self.loss = 0.0 # kick out these seeded ones first
        self.is_seed = is_seed

    def __len__(self):
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

    NUM_SEMANTIC_CLASSES = 2
    @staticmethod
    def _make_semantic(semantic_observation):
        wall  = np.isin(semantic_observation, obstacles)
        floor = np.isin(semantic_observation, walkable)

        return np.stack([wall, floor], axis=-1)

    def get_rgb_sequence(self):
        if not self.rgb_f:
            self.rgb_f = zarr.open(str(self.episode_dir / 'rgb'), mode='r')

        return self.rgb_f[:]

    def get_semantic_sequence(self):
        if not self.semantic_f:
            self.semantic_f = zarr.open(str(self.episode_dir / 'semantic'), mode='r')

        return self.semantic_f[:]

    def __getitem__(self, idx):
        rgb = 0
        if self.rgb:
            if not self.rgb_f:
                self.rgb_f = zarr.open(str(self.episode_dir / 'rgb'), mode='r')
            rgb = self.rgb_f[idx]

        """
        seg = 0
        if self.semantic:
            #_segs = torch.empty((len(episode), 256, 256, HabitatDataset.NUM_SEMANTIC_CLASSES))
            if not self.semantic_f:
                self.semantic_f = zarr.open(str(self.episode_dir / 'semantic'), mode='r')
            seg = HabitatDataset._make_semantic(self.semantic_f)
        """

        action = self.actions[idx]
        prev_action = self.actions[idx-1] if idx > 0 else torch.zeros_like(action)
        meta = self.meta[idx]

        """
        if self.augmentation:
            rgb, action, meta = self._aug(idx, rgb, action)
        """

        # rgb, mapview, segmentation, action, meta, episode, prev_action
        return rgb, 0, seg, action, meta, self.episode_idx, prev_action

    def __lt__(self, other):
        return self.loss < other.loss

    def _aug(self, start, rgb, action):
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
