import torch
import numpy as np


class LossSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, replay_buffer, batch_size):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.rng = np.random.default_rng()

    def __iter__(self):
        for _ in range(len(self)):
            idxs = self.replay_buffer.sample_k(self.batch_size, idxs_only=True)
            o = self.rng.permutation(self.batch_size)
            yield idxs[o]

    def __len__(self):
        # this doesn't guarantee that we hit all samples
        return len(self.replay_buffer) // self.batch_size


class ReplayBuffer(torch.utils.data.Dataset):

    def __init__(self, buffer_size, dshape, dtype):
        self.buffer_size = buffer_size
        self.idxs = torch.arange(self.buffer_size)
        self.size = 0

        self.targets = torch.empty((buffer_size, *dshape), dtype=dtype)
        self.goals = torch.empty((buffer_size, 2), dtype=torch.float32)
        self.prev_actions = torch.empty((buffer_size), dtype=torch.uint8)
        self.actions = torch.empty((buffer_size), dtype=torch.uint8)
        self.losses = np.empty((buffer_size), dtype=np.float32)

        self.rng = np.random.default_rng()

    def get_dataset(self):
        return torch.utils.data.TensorDataset(self.idxs, self.targets, self.goals, self.prev_actions, self.actions)

    def insert(self, target, goal, prev_action, action, loss=0.):
        if self.size >= self.buffer_size: # buffer full
            idx = self.losses[:self.size].argmin()
            if loss < self.losses[idx]:
                return
        else:
            idx = self.size
            self.size += 1

        self.targets[idx] = torch.as_tensor(target, dtype=self.targets.dtype)
        self.goals[idx] = torch.as_tensor(goal, dtype=torch.float32)
        self.prev_actions[idx] = prev_action
        self.actions[idx] = action
        self.losses[idx] = loss

        return idx

    def update_loss(self, idxs, losses):
        self.losses[idxs] = losses

    def sample_k(self, k, idxs_only=False):
        valid = self.losses[:self.size]
        if valid.sum() == 0: # uniform
            weights = np.ones((self.size)) / self.size
        else:
            weights = valid / valid.sum()

        idxs = self.rng.choice(self.idxs[:self.size], size=k, p=weights)
        if idxs_only:
            return idxs

        return idxs, self.targets[idxs], self.goals[idxs], self.prev_actions[idx], self.actions[idxs]

    def __getitem__(self, idx):
        if (idx > self.size).any():
            raise IndexError
        return idx, self.targets[idx], self.goals[idx], self.prev_actions[idx], self.actions[idx]

    def __len__(self):
        return self.size


