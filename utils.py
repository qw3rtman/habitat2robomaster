from itertools import repeat
import heapq
import shutil

import torch
import numpy as np


def repeater(loader):
    for loader in repeat(loader):
        for data in loader:
            yield data


def dataloader(data, batch_size, num_workers, collate_fn=None):
    return torch.utils.data.DataLoader(
            data, batch_size=batch_size, num_workers=num_workers,
            shuffle=True, drop_last=True, pin_memory=True,
            collate_fn=collate_fn)

def infinite_dataloader(data, batch_size, num_workers, collate_fn=None):
    return repeater(dataloader(data, batch_size, num_workers, collate_fn))


class StaticWrap(object):
    def __init__(self, data, batch_size, samples, num_workers, collate_fn=None):
        self.episodes = torch.utils.data.ConcatDataset(data)

        self.data = infinite_dataloader(self.episodes, batch_size, num_workers, collate_fn)
        self.samples = samples
        self.count = 0

    def __iter__(self):
        for i in range(self.samples):
            yield next(self.data)

    def __len__(self):
        return self.samples


class DynamicWrap(StaticWrap):
    def __init__(self, data, batch_size, samples, num_workers, capacity=1000):
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
            # we don't want to get back episode; this pops and then
            # pushes (as opposed to heappushpop)
            evicted_episode = heapq.heapreplace(self.episodes.datasets, episode)
            if not evicted_episode.is_seed:
                shutil.rmtree(evicted_episode.episode_dir, ignore_errors=True)
        else:
            heapq.heappush(self.episodes.datasets, episode)

    def post_dagger(self): # cleans cumulative_sizes and sets up dataloader for training
        self.episodes.cumulative_sizes = np.cumsum([len(episode) for episode in self.episodes.datasets])
        self.data = infinite_dataloader(self.episodes, self.batch_size, self.num_workers)

    def post_train(self): # clean heap; makes add_episode more efficient
        heapq.heapify(self.episodes.datasets)
