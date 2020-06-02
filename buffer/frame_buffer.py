import torch
import numpy as np

import zarr
from numcodecs import Blosc


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
        return len(self.replay_buffer) // self.batch_size


class ReplayBuffer(torch.utils.data.Dataset):

    def __init__(self, buffer_size, dshape, dtype, history_size=1, goal_size=3):
        buffer_size //= int(history_size)
        self.buffer_size = buffer_size
        self.history_size = int(history_size)

        self.dshape = dshape
        self.dtype = dtype

        self.idxs = torch.arange(self.buffer_size)
        self.size = 0
        self.uid = 0

        self.uids = np.empty((buffer_size), dtype=np.uint64)
        self.targets = torch.empty((buffer_size, history_size, *dshape), dtype=dtype)
        self.goals = torch.empty((buffer_size, goal_size), dtype=torch.float32)
        self.prev_actions = torch.empty((buffer_size), dtype=torch.uint8)
        self.actions = torch.empty((buffer_size), dtype=torch.uint8)
        self.losses = np.empty((buffer_size), dtype=np.float32)

        self.rng = np.random.default_rng()

    def get_dataset(self):
        return torch.utils.data.TensorDataset(self.idxs[:self.size], self.targets[:self.size], self.goals[:self.size], self.prev_actions[:self.size], self.actions[:self.size])

    def load(self, root):
        with open(root/'info.txt', 'r') as f:
            buffer_size, self.size, self.uid = map(int, f.readlines()[0].strip().split(' '))
        #assert buffer_size == self.buffer_size

        self.uids[:self.size] = np.load(root/'uids.npy')[:self.size]
        z = zarr.open(str(root/'targets'), mode='r')
        self.targets[:self.size] = torch.as_tensor(z[:self.size])
        self.goals[:self.size] = torch.load(root/'goals.pth')[:self.size]
        self.prev_actions[:self.size] = torch.load(root/'prev_actions.pth')[:self.size]
        self.actions[:self.size] = torch.load(root/'actions.pth')[:self.size]
        self.losses[:self.size] = torch.load(root/'losses.pth')[:self.size]

        print('[!] loaded buffer')

    def save(self, root, overwrite=False):
        try:
            root.mkdir(parents=True)
        except Exception:
            if not overwrite:
                print('[!] not overwriting!')
                return

        with open(root/'info.txt', 'w') as f:
            f.write(f'{self.buffer_size} {self.size} {self.uid}')

        np.save(root/'uids.npy', self.uids[:self.size])
        z = zarr.open(str(root/'targets'), mode='w', shape=(self.size, *self.targets.shape[1:]),
                chunks=(self.buffer_size//32, -1, -1, -1), dtype=self.targets.numpy().dtype,
                compressor=Blosc(cname='zstd', clevel=3))
        z[:] = self.targets[:self.size].numpy()

        torch.save(self.goals[:self.size], root/'goals.pth')
        torch.save(self.prev_actions[:self.size], root/'prev_actions.pth')
        torch.save(self.actions[:self.size], root/'actions.pth')
        torch.save(self.losses[:self.size], root/'losses.pth')

        print('[!] saved buffer')

    def insert(self, target, goal, prev_action, action, loss=0.):
        if self.size >= self.buffer_size: # buffer full
            # if loss is random, then this is random eviction
            idx = self.losses[:self.size].argmin()
            if loss < self.losses[idx]:
                return
        else:
            idx = self.size
            self.size += 1

        # no need to copy, indexing since makes a copy
        self.uids[idx] = self.uid
        self.targets[idx] = torch.as_tensor(target, dtype=self.targets.dtype)
        self.goals[idx] = torch.as_tensor(goal, dtype=torch.float32)
        self.prev_actions[idx] = prev_action
        self.actions[idx] = action
        self.losses[idx] = loss

        self.uid += 1
        return idx

    def update_loss(self, idxs, losses):
        self.losses[idxs] = losses

    def sample_k(self, k, idxs_only=False):
        valid = self.losses[:self.size]
        """ weighting scheme
        if valid.sum() == 0: # uniform
            weights = np.ones((self.size)) / self.size
        else:
            weights = valid / valid.sum()
        """

        idxs = self.rng.choice(self.idxs[:self.size], size=k)#, p=weights)
        if idxs_only:
            return idxs

        return idxs, self.targets[idxs].reshape(256, 256, -1), self.goals[idxs], self.prev_actions[idx], self.actions[idxs]

    def __getitem__(self, idx):
        if (idx > self.size).any():
            raise IndexError
        return idx, self.targets[idx], self.goals[idx], self.prev_actions[idx], self.actions[idx]

    def __len__(self):
        return self.size


