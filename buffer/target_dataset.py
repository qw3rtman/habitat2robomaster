import torch
import numpy as np
from joblib import Memory
import zarr

from pathlib import Path
from itertools import repeat
import time

from .util import cam_to_world, make_onehot, Wrap, C, rotate_origin_only, get_model_args, z2polar
import sys
sys.path.append('/u/nimit/Documents/robomaster/habitat2robomaster')
from model import GoalConditioned
from .goal_prediction import GoalPredictionModel

memory = Memory('/scratch/cluster/nimit/data/cache', mmap_mode='r+', verbose=0)
def get_dataset(source_teacher, goal_prediction, dataset_dir, scene, batch_size=128, num_workers=0, **kwargs):

    @memory.cache
    def get_episodes(source_teacher_path, goal_prediction_path, split_dir, dataset_size):
        episode_dirs = list(split_dir.iterdir())
        num_episodes = int(max(1, dataset_size * len(episode_dirs)))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # set up source teacher
        student_args = get_model_args(source_teacher_path, 'student_args')
        data_args = get_model_args(source_teacher_path, 'data_args')
        source_teacher = GoalConditioned(**student_args, **data_args).to(device)
        source_teacher.load_state_dict(torch.load(source_teacher_path, map_location=device))
        source_teacher.eval()

        # set up goal prediction
        model_args = get_model_args(goal_prediction_path, 'model_args')
        data_args = get_model_args(goal_prediction_path, 'data_args')
        goal_prediction = GoalPredictionModel(**model_args).to(device)
        goal_prediction.load_state_dict(torch.load(goal_prediction_path, map_location=device))
        goal_prediction.eval()

        data = []
        for i, episode_dir in enumerate(episode_dirs[:num_episodes]):
            data.append(TargetDataset(source_teacher, goal_prediction, episode_dir,
                scene, data_args['zoom'], data_args['steps']))

            if i % 100 == 0:
                print(f'[{i:05}/{num_episodes}]')

        return data

    def make_dataset(is_train):
        split = 'train' if is_train else 'val'

        start = time.time()
        data = get_episodes(
                Path(source_teacher),
                Path(goal_prediction),
                Path(dataset_dir) / split,
                kwargs.get('dataset_size', 1.0))
        print(f'{split}: {len(data)} episodes in {time.time()-start:.2f}s')

        return Wrap(data, batch_size, 1000 if is_train else 100, num_workers)

    return make_dataset(True), make_dataset(False)

class TargetDataset(torch.utils.data.Dataset):
    def __init__(self, source_teacher, goal_prediction, episode_dir, scene, zoom, steps):
        self.episode_dir = episode_dir
        self.scene = scene
        self.zoom = zoom
        self.steps = steps

        self.rgb_f = zarr.open(str(self.episode_dir / 'rgb'), mode='r')

        source_teacher.eval()
        goal_prediction.eval()

        onehot = make_onehot(np.uint8(zarr.open(str(self.episode_dir / 'semantic'), mode='r')[:]), scene=scene)
        onehot = torch.as_tensor(onehot.reshape(-1, C, 160, 384), dtype=torch.float).cuda()
        self.waypoints = torch.empty(self.rgb_f.shape[0], 4, steps, 2)
        with torch.no_grad():
            for a in range(4):
                self.waypoints[:, a] = self.zoom * goal_prediction(onehot,
                        a*torch.ones(self.rgb_f.shape[0]).cuda()).cpu()
                print(onehot[5].sum(), episode_dir, self.waypoints[5, a])
                print()
                print()
                print()
                print()

        r, t = z2polar(*self.waypoints.numpy().reshape(2, -1, 4, steps))
        r, t = torch.as_tensor(r[-1]-r), torch.as_tensor((t[-1]-t)-(np.pi/2))
        #print(episode_dir, r[0,0])

        self.goal = torch.stack([r, torch.cos(-t), torch.sin(-t)], dim=-1)
        goal = self.goal.cuda()
        self.actions = torch.empty(self.goal.shape[:3], dtype=torch.long)
        onehot = onehot.reshape(-1, 160, 384, C)
        with torch.no_grad():
            for a in range(4):
                for t in range(steps):
                    self.actions[:,a,t] = source_teacher((onehot, goal[:,a,t])).sample().squeeze().cpu()
                    #print(self.actions[:,a,t], self.goal[:,a,t,0])

    def __len__(self):
        return self.rgb_f.shape[0] * 4 * self.steps

    def __getitem__(self, idx):
        rgb = self.rgb_f[idx//20]
        goal = self.goal[idx//20][idx%4][idx%self.steps]
        action = self.actions[idx//20][idx%4][idx%self.steps]
        waypoints = self.waypoints[idx//20][idx%4]
        waypoint_idx = idx % self.steps

        return rgb, goal, action, waypoints, waypoint_idx
