from itertools import repeat
import heapq
import shutil

import torch
import numpy as np


def repeater(loader):
    for loader in repeat(loader):
        for data in loader:
            yield data


def dataloader(data, batch_size, num_workers):
    return torch.utils.data.DataLoader(
            data, batch_size=batch_size, num_workers=num_workers,
            shuffle=True, drop_last=True, pin_memory=True)


def infinite_dataloader(data, batch_size, num_workers):
    return repeater(dataloader(data, batch_size, num_workers))


class StaticWrap(object):
    def __init__(self, data, batch_size, samples, num_workers):
        self.episodes = torch.utils.data.ConcatDataset(data)

        self.data = infinite_dataloader(self.episodes, batch_size, num_workers)
        self.samples = samples
        self.count = 0

    def __iter__(self):
        for i in range(self.samples):
            yield next(self.data)

    def __len__(self):
        return self.samples


class DynamicWrap(StaticWrap):
    def __init__(self, data, batch_size, samples, num_workers, capacity=2000):
        """
        capacity : capacity of rollout episodes; true capacity is len(seeded) + capacity
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.episodes = data
        if len(self.episodes) > 0:
            self.episodes = torch.utils.data.ConcatDataset(self.episodes)
            self.data = infinite_dataloader(self.episodes, batch_size, num_workers)

        self.samples = samples
        self.count = 0

        self.capacity = capacity

    def add_episode(self, episode):
        if len(self.episodes) == 0:
            self.episodes = torch.utils.data.ConcatDataset([episode])
            return

        if len(self.episodes.datasets) >= self.capacity:
            evicted_episode = heapq.heappushpop(self.episodes.datasets, episode)
            if not evicted_episode.is_seed:
                shutil.rmtree(evicted_episode.episode_dir, ignore_errors=True)
        else:
            heapq.heappush(self.episodes.datasets, episode)

    def post_dagger(self): # cleans cumulative_sizes and sets up dataloader for training
        self.episodes.cumulative_sizes = np.cumsum([len(episode) for episode in self.episodes.datasets])
        self.data = infinite_dataloader(self.episodes, self.batch_size, self.num_workers)

    def post_train(self): # clean heap; makes add_episode more efficient
        heapq.heapify(self.episodes.datasets)
