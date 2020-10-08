import torch
import numpy as np

from .dataset import Wrap, make_onehot


class SingleDataset(torch.utils.data.Dataset):

    def __init__(self, frames, xy, actions, scene):
        self.frames = frames
        self.xy = xy
        self.actions = np.uint8(actions)
        self.scene = scene

    def __len__(self):
        return self.actions.shape[0]-1

    def __getitem__(self, idx):
        """
        t1, t2 = make_onehot(np.stack([
            self.frames[idx], self.frames[idx+t]
        ]), scene=self.scene)
        """

        return self.frames[idx], self.xy[idx], self.actions[idx]-1


class PairDataset(torch.utils.data.Dataset):

    def __init__(self, frames, xy, actions, scene, temporal_dim=1):
        self.frames = frames
        self.xy = xy
        self.actions = np.uint8(actions)
        self.scene = scene
        self.t_min, self.t_max = 1, temporal_dim+1

    def __len__(self):
        return self.actions.shape[0]-1

    def __getitem__(self, idx):
        t = np.random.randint(self.t_min, self.t_max)
        if idx+t > len(self)-1:
            t = len(self)-idx

        """
        t1, t2 = make_onehot(np.stack([
            self.frames[idx], self.frames[idx+t]
        ]), scene=self.scene)
        """

        t1, t2 = self.frames[idx], self.frames[idx+t]
        return t1, t2, self.actions[idx]-1, t-1, xy


class ReplayBuffer():

    def __init__(self, scene):
        self.scene = scene

        self.frames = []
        self.actions = []
        self.xy = []
        self.length = 0

    def get_dataset(self, iterations=50, batch_size=128, num_workers=0, temporal_dim=1):
        """
        data = [PairDataset(frames[:self.length] if i == len(self.frames)-1 else frames, \
                            xy[:self.length] if i == len(self.frames)-1 else xy, \
                            actions, self.scene, temporal_dim=temporal_dim) \
            for i, (frames, xy, actions) in enumerate(zip(self.frames, self.xy, self.actions)) \
        if len(actions) > 0]
        """
        data = [SingleDataset(frames[:self.length] if i == len(self.frames)-1 else frames, \
                            xy[:self.length] if i == len(self.frames)-1 else xy, \
                            actions, self.scene) \
            for i, (frames, xy, actions) in enumerate(zip(self.frames, self.xy, self.actions)) \
        if len(actions) > 0]

        return Wrap(data, batch_size, iterations, num_workers)

    def new_episode(self):
        if len(self.frames) > 0:
            self.frames[-1] = self.frames[-1][:self.length]
            self.xy[-1] = self.xy[-1][:self.length]

        self.frames.append(np.empty((500, 160, 384, 3), dtype=np.uint8))
        self.xy.append(np.empty((500, 2), dtype=np.float32))
        self.actions.append([])

        self.length = 0

    def insert(self, frame, action, xy):
        self.frames[-1][self.length] = frame
        self.xy[-1][self.length] = xy
        self.actions[-1].append(action)
        self.length += 1

    def __len__(self):
        return sum([len(action) for action in self.actions[:-1]]) + self.length
