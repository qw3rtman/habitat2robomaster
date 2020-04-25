import numpy as np
import torch
import cv2
import pandas as pd
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from PIL import Image
from pyquaternion import Quaternion

from pathlib import Path
import argparse
from operator import itemgetter
from itertools import repeat

from utils import StaticWrap, DynamicWrap

ACTIONS = torch.eye(4)


def get_dataset(dataset_dir, dagger=False, interpolate=False, rnn=False, capacity=2000, batch_size=128, num_workers=4, augmentation=False, **kwargs):

    def make_dataset(is_train):
        data = list()
        train_or_val = 'train' if is_train else 'val'

        episodes = list((Path(dataset_dir) / train_or_val).iterdir())
        num_episodes = int(max(1, kwargs.get('dataset_size', 1.0) * len(episodes)))

        for episode_dir in episodes[:num_episodes]:
            data.append(HabitatDataset(episode_dir, is_seed=dagger, interpolate=interpolate if is_train else True, augmentation=augmentation))

        print('%s: %d' % (train_or_val, len(data)))

        if dagger:
            return DynamicWrap(data, batch_size, 100000 if is_train else 10000, num_workers, capacity=capacity)                    # samples == # steps
        elif rnn:
            return StaticWrap(EpisodeDataset(data), batch_size, 500 if is_train else 50, num_workers, collate_fn=collate_episodes) # samples == episodes
        else:
            return StaticWrap(data, batch_size, 25000 if is_train else 2500, num_workers)                                        # samples == # steps

    return make_dataset(True), make_dataset(False)


def collate_episodes(episodes):
    rgbs, segs, actions, prev_actions, metas = [], [], [], [], []
    for i, episode in enumerate(episodes):
        rgbs.append(torch.zeros((len(episode), 256, 256, 3), dtype=torch.uint8))
        segs.append(torch.zeros((len(episode), 256, 256, 1), dtype=torch.uint8))

        actions.append(episode.actions)
        prev_actions.append(torch.zeros(len(episode), dtype=torch.long))
        prev_actions[i][1:] = episode.actions[:-1].clone()

        metas.append(torch.zeros((len(episode), 2), dtype=torch.float))

        for t, step in enumerate(episode):
            rgb, _, seg, action, meta, _, prev_action = step
            rgbs[i][t] = rgb
            segs[i][t] = seg
            metas[i][t] = meta

    rgb_batch = pad_sequence(rgbs)
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
    def __init__(self, episode_dir, is_seed=False, interpolate=False, augmentation=False):
        if not isinstance(episode_dir, Path):
            episode_dir = Path(episode_dir)

        self.episode_idx = 0
        self.loss = 0.0 # kick out these seeded ones first
        self.is_seed = is_seed
        self.augmentation = augmentation

        self.episode_dir = episode_dir

        self.measurements = pd.read_csv(episode_dir / 'episode.csv')

        _indices = np.ones(len(self.measurements), dtype=np.bool)
        if interpolate:
            for i in range(len(_indices)):
                if i % 5 in [0, 1, 4] and i + 1 < len(_indices):
                    _indices[i] = True
                else:
                    _indices[i] = False

            self.imgs = [rgb for i, rgb in enumerate(sorted(episode_dir.glob('rgb_*.png'))) if _indices[i]]
            self.segs = [seg for i, seg in enumerate(sorted(episode_dir.glob('seg_*.npy'))) if _indices[i]]
        else:
            self.imgs = list(sorted(episode_dir.glob('rgb_*.png')))
            self.segs = list(sorted(episode_dir.glob('seg_*.npy')))

        self.compass = torch.Tensor(np.stack(itemgetter('compass_r','compass_t')(self.measurements), -1)[_indices])
        self.positions = torch.Tensor(np.stack(itemgetter('x','y','z')(self.measurements), -1)[_indices])
        self.rotations = torch.Tensor(np.stack(itemgetter('i','j','k','l')(self.measurements), -1)[_indices])

        self.left = Quaternion(axis=[0,1,0], degrees=10)
        self.right = Quaternion(axis=[0,1,0], degrees=-10)

        if interpolate:
            _action_indices = []
            for _index, include in enumerate(_indices):
                if not include:
                    continue
                if _index % 5 == 0:
                    _action_indices.append(_index)
                if _index % 5 == 1:
                    _action_indices.append(_index - 1)
                if _index % 5 == 4:
                    _action_indices.append(_index + 1)

            self.actions = torch.LongTensor(self.measurements['action'])[_action_indices]
            self.measurements = self.measurements[_indices]
        else:
            self.actions = torch.LongTensor(self.measurements['action'])

        self.info = pd.read_csv(episode_dir / 'info.csv').iloc[0]
        #self.scene = self.info['scene']
        self.start_position = torch.Tensor(itemgetter('start_pos_x', 'start_pos_y', 'start_pos_z')(self.info))
        # NOTE: really a quaternion
        self.start_rotation = torch.Tensor(itemgetter('start_rot_i', 'start_rot_j', 'start_rot_k', 'start_rot_l')(self.info))
        self.end_position   = torch.Tensor(itemgetter('end_pos_x', 'end_pos_y', 'end_pos_z')(self.info))

    def __len__(self):
        return len(self.imgs)

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

    def __getitem__(self, idx):
        rgb    = Image.open(self.imgs[idx])
        seg    = torch.Tensor(np.load(self.segs[idx])==2).unsqueeze(dim=-1)

        action = self.actions[idx]
        prev_action = self.actions[idx-1] if idx > 0 else torch.zeros_like(action)

        if self.augmentation:
            rgb, action, meta = self.aug(idx, rgb, action)
        else:
            meta = self._get_direction(idx)

        rgb  = torch.Tensor(np.uint8(rgb))

        # rgb, mapview, segmentation, action, meta, episode, prev_action
        return rgb, None, seg, action, meta, self.episode_idx, prev_action

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
