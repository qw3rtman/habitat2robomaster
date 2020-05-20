import numpy as np
import torch
import cv2
import pandas as pd
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from rad import data_augs
import imgaug.augmenters as iaa
from pyquaternion import Quaternion
from pathlib import Path
from joblib import Memory
import zarr

import argparse
from operator import itemgetter
from itertools import repeat
from collections import defaultdict, OrderedDict
import subprocess
import random
import time

from utils import StaticWrap, DynamicWrap

ACTIONS = torch.eye(4)
NUM_EPISODES_DAGGER = 25

memory = Memory('/scratch/cluster/nimit/data/cache', mmap_mode='r+', verbose=0)
def get_dataset(dataset_dir, seed=False, interpolate=False, rnn=False, capacity=2000, batch_size=128, num_workers=0, augmentation=False, depth=False, rgb=True, semantic=False, **kwargs):

    #@memory.cache
    def get_episodes(split_dir):
        episode_dirs = list(split_dir.iterdir())
        num_episodes = int(max(1, kwargs.get('dataset_size', 1.0) * len(episode_dirs)))

        data = []
        for i, episode_dir in enumerate(episode_dirs[:num_episodes]):
            episode = HabitatDataset(episode_dir, interpolate=interpolate, augmentation=augmentation, depth=depth, rgb=rgb, semantic=semantic)
            # bounds memory usage by (250 x B x 256 x 256 x 3) x 4 bytes
            # i.e: 0.20 GB per batch dimension
            if len(episode) <= 250:
                data.append(episode)

            if i % 500 == 0:
                print(f'[{i:05}/{num_episodes}]')

        return data

    @memory.cache
    def get_seed_episodes(split_dir):
        # pick 5 episodes from each scene as starter
        # (332+90)*120*5 = ~250k
        episode_dirs = list(split_dir.iterdir())

        freq = defaultdict(int)
        data = []
        for i, episode_dir in enumerate(episode_dirs):
            scene = episode_dir.stem.split('-')[0]
            if freq[scene] == NUM_EPISODES_DAGGER:
                continue
            episode = HabitatDataset(episode_dir, interpolate=interpolate, augmentation=augmentation, depth=depth, rgb=rgb, semantic=semantic)

            # bounds memory usage by (250 x B x 256 x 256 x 3) x 4 bytes
            # i.e: 0.20 GB per batch dimension
            if len(episode) <= 250:
                freq[scene] += 1
                data.append(episode)

            if sum(freq.values()) == NUM_EPISODES_DAGGER * (332+90):
                break

            if i % 500 == 0:
                print(f'[{sum(freq.values()):04}/{NUM_EPISODES_DAGGER}]')

        return data

    def make_dataset(is_train):
        split = 'train' if is_train else 'val'

        start = time.time()
        if seed:
            data = get_seed_episodes(Path(dataset_dir) / split)
        else:
            data = get_episodes(Path(dataset_dir) / split)
        print(f'{split}: {len(data)} in {time.time()-start:.2f}s')

        if rnn:
            bucket_batch_sampler = BucketBatchSampler(data, batch_size)

            num_episodes = 512 if is_train else 32
            if not seed:
                num_episodes = 256 if is_train else 16 # should be < NUM_SCENES*EPISODES_PER_SCENE
            return StaticWrap(EpisodeDataset(data), batch_size, num_episodes, num_workers, collate_fn=collate_episodes, batch_sampler=bucket_batch_sampler) # samples == episodes
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

        idx, self.batches, self.length_idx = 0, [0] * len(episodes), {}
        for length, indices in buckets.items():
            random.shuffle(indices)
            self.batches[idx:idx+len(indices)] = indices
            self.length_idx[length] = idx
            idx += len(indices)

        idx = 0
        for length in range(0, 251):
            if length not in self.length_idx:
                self.length_idx[length] = idx
            else:
                idx = self.length_idx[length]

    def __len__(self):
        return len(self.batches) // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            start = len(self.batches)
            while start + self.batch_size >= len(self.batches):
                length = np.random.randint(250) # uniformly sample episode lengths
                start = np.random.randint(self.length_idx[length], len(self.batches))

            batch = self.batches[start:start+self.batch_size]
            random.shuffle(batch)

            yield batch

random_crop = iaa.KeepSizeByResize(iaa.Crop((10, 40), keep_size=False), interpolation="cubic")
def collate_episodes(episodes):
    lengths = np.fromiter(map(len, episodes), dtype=np.uint8)
    length = np.random.randint(*(np.array([0.5, 0.8]) * min(lengths)))
    start = (np.random.random(size=len(episodes)) * (lengths-length)).astype(np.uint8)
    end = start + length

    depths, rgbs, segs, actions, prev_actions, metas = [], [], [], [], [], []
    for i, episode in enumerate(episodes):
        _meta = episode.meta.clone().detach()
        _actions = episode.actions.clone().detach()

        # encourages augmentation in goal specification
        temporal_aug = np.random.random() < 0.00
        if temporal_aug:
            _meta = _meta[start[i]:end[i]] - _meta[end[i]]
            _actions = _actions[start[i]:end[i]]

        # visual augmentation
        flip_aug = np.random.random() < 0.0
        if flip_aug:
            _meta = np.multiply(_meta, np.array([-1., 1.]), dtype=np.float32)
            left, right = _actions == 2, _actions == 3
            _actions[left], _actions[right] = 3, 2 # left <-> right

        #metas.append(episode.compass)
        metas.append(_meta)
        actions.append(_actions)

        _prev_actions = torch.empty_like(actions[-1])
        _prev_actions[0] = 0
        _prev_actions[1:].copy_(actions[-1][:-1])
        prev_actions.append(_prev_actions)

        if episode.depth:
            depth_sequence = episode.get_depth_sequence().reshape(-1, 256, 256, 1)
            if temporal_aug:
                depth_sequence = depth_sequence[start[i]:end[i]]

            if flip_aug:
                depth_sequence = torch.as_tensor(depth_sequence).flip(dims=(1,))

            depths.append(torch.as_tensor(depth_sequence).type(torch.float))

        if episode.rgb:
            rgb_sequence = episode.get_rgb_sequence()
            if temporal_aug:
                rgb_sequence = rgb_sequence[start[i]:end[i]]

            # visual augmentation (https://arxiv.org/pdf/2004.14990.pdf)
            # input: numpy T x H x W x C; i.e: T x 256 x 256 x 3
            # output: torch, same dims
            p = np.random.uniform(0., 1.)
            if p < 0.10: # random crop; ratio from RAD paper
                #print('random crop')
                rgb_sequence = random_crop(images=rgb_sequence)
            elif p < 0.10: # random cutout; ratio from RAD paper
                #print('random cutout')
                rgb_sequence = data_augs.random_cutout(rgb_sequence.transpose(0,3,1,2), min_cut=25, max_cut=76).transpose(0,2,3,1)
            elif p < 0.10: # random cutout color; ratio from RAD paper
                #print('random cutout color')
                rgb_sequence = data_augs.random_cutout_color(rgb_sequence.transpose(0,3,1,2), min_cut=25, max_cut=76).transpose(0,2,3,1)

            p = np.random.uniform(0., 1.)
            if p < 0.00: # random grayscale
                #print('random grayscale')
                rgb_sequence = 255.*data_augs.random_grayscale(torch.as_tensor(rgb_sequence.transpose(0,3,1,2)/255.), p=1.0).permute(0,2,3,1)
            elif p < 0.00: # color aug; slow, but apparently important
                #print('color jitter')
                rgb_sequence = 255*data_augs.random_color_jitter(torch.as_tensor(rgb_sequence.transpose(0,3,1,2)/255., dtype=torch.float)).permute(0,2,3,1)

            if flip_aug:
                rgb_sequence = torch.as_tensor(rgb_sequence).flip(dims=(2,))

            rgbs.append(torch.as_tensor(rgb_sequence).type(torch.uint8))

        if episode.semantic:
            segs.append(torch.as_tensor(episode.get_semantic_sequence()).type(torch.uint8))
            # TODO: if temporal, slice the semantics
            # TODO: if flip, flip the semantics

        #worker_info = torch.utils.data.get_worker_info()
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

    return depths, rgbs, segs, action_batch, prev_action_batch, meta_batch, mask


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
    def __init__(self, episode_dir, interpolate=False, augmentation=False, depth=False, rgb=True, semantic=False):
        if not isinstance(episode_dir, Path):
            episode_dir = Path(episode_dir)

        self.episode_dir = episode_dir
        self.depth, self.rgb, self.semantic = depth, rgb, semantic

        self.interpolate = interpolate

        with open(self.episode_dir / 'episode.csv', 'r') as f:
            measurements = f.readlines()[1:]
        x = np.genfromtxt(measurements, delimiter=',', dtype=np.float32)
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        self.positions = torch.as_tensor(x[:, 5:8])
        self.rotations = torch.as_tensor(x[:, 8:])
        self.actions = torch.LongTensor(x[:, 1])
        self.compass = torch.as_tensor(x[:, 3:5])

        with open(self.episode_dir / 'info.csv', 'r') as f:
            info = f.readlines()[1].split(',')
        self.start_position = torch.Tensor(list(map(float, info[8:11])))
        self.start_rotation = torch.Tensor(list(map(float, info[11:15])))
        self.end_position   = torch.Tensor(list(map(float, info[15:18])))

        self.meta = torch.empty((self.actions.shape[0], 2))
        for i in range(self.actions.shape[0]):
            self.meta[i] = self._get_direction(i)

        if self.depth:
            assert (self.episode_dir / 'depth').exists()
            self.depth_f = zarr.open(str(self.episode_dir / 'depth'), mode='r')

        if self.rgb:
            assert (self.episode_dir / 'rgb').exists()
            self.rgb_f = zarr.open(str(self.episode_dir / 'rgb'), mode='r')

        if self.semantic:
            assert (self.episode_dir / 'semantic').exists()
            self.semantic_f = zarr.open(str(self.episode_dir / 'semantic'), mode='r')

        # DAgger
        self.episode_idx = 0

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

    def get_depth_sequence(self):
        if not self.depth_f:
            self.depth_f = zarr.open(str(self.episode_dir / 'depth'), mode='r')

        return self.depth_f[:]

    def get_rgb_sequence(self):
        if not self.rgb_f:
            self.rgb_f = zarr.open(str(self.episode_dir / 'rgb'), mode='r')

        return self.rgb_f[:]

    NUM_SEMANTIC_CLASSES = 10
    top10 = np.array([1, 2, 17, 4, 40, 0, 9, 7, 5, 3])

    @staticmethod
    def make_onehot(semantic):
        onehot = torch.zeros(*semantic.shape, HabitatDataset.NUM_SEMANTIC_CLASSES, dtype=torch.long)
        for idx, _class in enumerate(HabitatDataset.top10):
            onehot[..., idx] = torch.as_tensor(semantic == _class)
        return onehot

    def get_semantic_sequence(self):
        if not self.semantic_f:
            self.semantic_f = zarr.open(str(self.episode_dir / 'semantic'), mode='r')
        return HabitatDataset.make_onehot(self.semantic_f[:])

    def __getitem__(self, idx):
        rgb = 0
        if self.rgb:
            if not self.rgb_f:
                self.rgb_f = zarr.open(str(self.episode_dir / 'rgb'), mode='r')
            rgb = self.rgb_f[idx]

        seg = 0
        if self.semantic:
            if not self.semantic_f:
                self.semantic_f = zarr.open(str(self.episode_dir / 'semantic'), mode='r')
            seg = HabitatDataset.make_semantic(self.semantic_f[idx])

        action = self.actions[idx]
        prev_action = self.actions[idx-1] if idx > 0 else torch.zeros_like(action)
        meta = self.meta[idx]
        #meta = self.compass[idx]

        # rgb, mapview, segmentation, action, meta, episode, prev_action
        return rgb, 0, seg, action, meta, self.episode_idx, prev_action

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
