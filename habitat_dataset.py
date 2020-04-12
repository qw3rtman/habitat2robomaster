import numpy as np
import torch
import cv2
import pandas as pd
from torchvision import transforms
from PIL import Image

from pathlib import Path
import argparse
from operator import itemgetter
from itertools import repeat
import quaternion

from utils import StaticWrap, DynamicWrap

ACTIONS = torch.eye(4)


def get_dataset(dataset_dir, dagger=False, interpolate=False, capacity=2000, batch_size=128, num_workers=4, **kwargs):

    def make_dataset(is_train):
        data = list()
        train_or_val = 'train' if is_train else 'val'

        episodes = list((Path(dataset_dir) / train_or_val).iterdir())
        num_episodes = int(max(1, kwargs.get('dataset_size', 1.0) * len(episodes)))

        for episode_dir in episodes[:num_episodes]:
            data.append(HabitatDataset(episode_dir, is_seed=dagger, interpolate=interpolate if is_train else True))

        print('%s: %d' % (train_or_val, len(data)))

        if dagger:
            return DynamicWrap(data, batch_size, 1000 if is_train else 100, num_workers, capacity=capacity)
        else:
            return StaticWrap(data, batch_size, 1000 if is_train else 100, num_workers)

    return make_dataset(True), make_dataset(False)


class HabitatDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir, is_seed=False, interpolate=False):
        if not isinstance(episode_dir, Path):
            episode_dir = Path(episode_dir)

        self.episode_idx = 0
        self.loss = 0.0 # kick out these seeded ones first
        self.is_seed = is_seed

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
            self.segs = [seg for i, rgb in enumerate(sorted(episode_dir.glob('seg_*.npy'))) if _indices[i]]
        else:
            self.imgs = list(sorted(episode_dir.glob('rgb_*.png')))
            self.segs = list(sorted(episode_dir.glob('seg_*.npy')))

        self.positions = torch.Tensor(np.stack(itemgetter('x','y','y')(self.measurements), -1)[_indices])
        self.rotations = torch.Tensor(np.stack(itemgetter('i','j','k','l')(self.measurements), -1)[_indices])

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
        self.start_position = torch.Tensor(itemgetter('start_pos_x', 'start_pos_y', 'start_pos_z')(self.info))
        # NOTE: really a quaternion
        self.start_rotation = torch.Tensor(itemgetter('start_rot_i', 'start_rot_j', 'start_rot_k', 'start_rot_l')(self.info))
        self.end_position   = torch.Tensor(itemgetter('end_pos_x', 'end_pos_y', 'end_pos_z')(self.info))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        rgb    = Image.open(self.imgs[idx])
        rgb    = torch.Tensor(np.uint8(rgb))

        #seg    = torch.Tensor(np.float32(np.load(self.segs[idx])))
        action = self.actions[idx]

        # curr rot, end pos - curr pos
        meta = torch.cat([self.rotations[idx,[0,2]], self.end_position[:2] - self.positions[idx,:2]], dim=-1)

        # rgb, mapview, segmentation, action, meta, episode
        return rgb, 0, 0, action, meta, self.episode_idx

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
