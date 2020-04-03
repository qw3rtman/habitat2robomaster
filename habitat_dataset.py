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

from utils import Wrap

ACTIONS = torch.eye(5)


def get_dataset(dataset_dir, batch_size=128, num_workers=4, **kwargs):

    def make_dataset(is_train):
        data = list()
        train_or_val = 'train' if is_train else 'val'

        for episode_dir in (Path(dataset_dir) / train_or_val).iterdir():
            data.append(HabitatDataset(episode_dir))

        data = torch.utils.data.ConcatDataset(data)

        print('%s: %d' % (train_or_val, len(data)))

        return Wrap(data, batch_size, 1000 if is_train else 100, num_workers)

    train_dataset = make_dataset(True)
    test_dataset = make_dataset(False)

    return train_dataset, test_dataset


class HabitatDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir):
        if not isinstance(episode_dir, Path):
            episode_dir = Path(episode_dir)

        self.episode_dir = episode_dir

        self.imgs = list(sorted(episode_dir.glob('rgb_*.png')))
        self.segs = list(sorted(episode_dir.glob('seg_*.npy')))

        self.measurements = pd.read_csv(episode_dir / 'episode.csv')
        self.positions = np.stack(itemgetter('x','y','y')(self.measurements), -1)
        self.rotations = np.stack(itemgetter('i','j','k','l')(self.measurements), -1)
        self.actions = np.array(self.measurements['action'])

        self.info = pd.read_csv(episode_dir / 'info.csv').iloc[0]
        self.start_position = np.array([itemgetter('start_pos_x', 'start_pos_y', 'start_pos_z')(self.info)])
        # NOTE: really a quaternion
        self.start_rotation = np.array([itemgetter('start_rot_i', 'start_rot_j', 'start_rot_k', 'start_rot_l')(self.info)])
        self.end_position   = np.array([itemgetter('end_pos_x', 'end_pos_y', 'end_pos_z')(self.info)])

        self.transform  = transforms.ToTensor()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        rgb    = self.transform(Image.open(self.imgs[idx]))
        seg    = torch.Tensor(np.float32(np.load(self.segs[idx])))
        action = ACTIONS[self.actions[idx]].clone()

        # rgb, mapview, segmentation, action, debug
        return rgb, 0, seg, action, (self.start_position, self.start_rotation, self.end_position)


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
