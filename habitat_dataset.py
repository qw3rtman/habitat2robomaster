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


def get_dataset(dataset_dir, dagger=False, capacity=2000, batch_size=128, num_workers=4, **kwargs):

    def make_dataset(is_train):
        data = list()
        train_or_val = 'train' if is_train else 'val'

        episodes = list((Path(dataset_dir) / train_or_val).iterdir())
        num_episodes = int(max(1, kwargs.get('dataset_size', 1.0) * len(episodes)))

        for episode_dir in episodes[:num_episodes]:
            data.append(HabitatDataset(episode_dir, is_seed=dagger))

        print('%s: %d' % (train_or_val, len(data)))

        if dagger:
            return DynamicWrap(data, batch_size, 10, num_workers, capacity=capacity)
        else:
            return StaticWrap(data, batch_size, 10 if is_train else 100, num_workers)

    return make_dataset(True), None if dagger else make_dataset(False)


class HabitatDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir, is_seed=False):
        if not isinstance(episode_dir, Path):
            episode_dir = Path(episode_dir)

        self.episode_idx = 0
        self.loss = 0.0 # kick out these seeded ones first
        self.is_seed = is_seed

        self.episode_dir = episode_dir

        self.imgs = list(sorted(episode_dir.glob('rgb_*.png')))
        self.segs = list(sorted(episode_dir.glob('seg_*.npy')))

        self.measurements = pd.read_csv(episode_dir / 'episode.csv')
        self.positions = np.stack(itemgetter('x','y','y')(self.measurements), -1)
        self.rotations = np.stack(itemgetter('i','j','k','l')(self.measurements), -1)
        self.actions = np.array(self.measurements['action'])

        self.info = pd.read_csv(episode_dir / 'info.csv').iloc[0]
        self.start_position = torch.Tensor(itemgetter('start_pos_x', 'start_pos_y', 'start_pos_z')(self.info))
        # NOTE: really a quaternion
        self.start_rotation = torch.Tensor(itemgetter('start_rot_i', 'start_rot_j', 'start_rot_k', 'start_rot_l')(self.info))
        self.end_position   = torch.Tensor(itemgetter('end_pos_x', 'end_pos_y', 'end_pos_z')(self.info))

        self.transform  = transforms.ToTensor()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        rgb    = Image.open(self.imgs[idx])
        rgb    = torch.Tensor(np.uint8(rgb))

        #seg    = torch.Tensor(np.float32(np.load(self.segs[idx])))
        action = ACTIONS[self.actions[idx]].clone()
        meta   = torch.cat([self.start_position, self.start_rotation, self.end_position], dim=-1)

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
