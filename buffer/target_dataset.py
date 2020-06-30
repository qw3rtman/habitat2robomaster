import torch
import numpy as np
from joblib import Memory
import zarr
import pandas as pd

from pathlib import Path
from itertools import repeat
import time

from .util import cam_to_world, make_onehot, get_navigable, Wrap, C, rotate_origin_only, get_model_args
import sys
sys.path.append('/u/nimit/Documents/robomaster/habitat2robomaster')
from model import GoalConditioned
from .goal_prediction import GoalPredictionModel

ACTIONS = ['S', 'F', 'L', 'R']

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

        floor = np.asarray((make_onehot(np.uint8(zarr.open(str(self.episode_dir
            /'semantic'), mode='r')[:]), scene=scene)), dtype=np.bool).squeeze()
        navigable = get_navigable(pd.read_csv(self.episode_dir/'episode.csv'), scene)

        print(floor.dtype, navigable.dtype)
        onehot = torch.as_tensor(floor & navigable,
                dtype=torch.float).reshape(-1, C, 160, 384).cuda()

        self.waypoints = torch.empty(self.rgb_f.shape[0], 4, steps, 2)
        #print(episode_dir, onehot.sum())
        with torch.no_grad():
            for a in range(4):
                self.waypoints[:, a] = goal_prediction(onehot,
                        a*torch.ones(self.rgb_f.shape[0]).cuda()).cpu()
                #print(self.waypoints[5, a])

        self.r = np.sqrt(np.square(self.zoom*self.waypoints[...,0]) +
                np.square(self.zoom*self.waypoints[...,1]))
        self.t = np.arctan2(-self.waypoints[...,0], self.waypoints[...,1])

        self.goal = torch.as_tensor(np.stack([self.r, np.cos(-self.t), np.sin(-self.t)], axis=-1))
        goal = self.goal.cuda()
        self.actions = torch.empty(self.goal.shape[:3], dtype=torch.long)
        onehot = onehot.reshape(-1, 160, 384, C)
        with torch.no_grad():
            for a in range(4):
                for t in range(steps):
                    self.actions[:,a,t] = source_teacher((onehot, goal[:,a,t])).sample().squeeze().cpu()
                    #print(self.actions[:,a,t], self.goal[:,a,t,0])

        self.onehot = onehot.cpu() # TODO: remove after debugging

    def __len__(self):
        return self.rgb_f.shape[0] * 4 * self.steps

    def __getitem__(self, idx):
        rgb = self.rgb_f[idx//(4*self.steps)]
        goal = self.goal[idx//(4*self.steps)][idx%4][idx%self.steps]
        r = self.r[idx//(4*self.steps)][idx%4][idx%self.steps]
        t = self.t[idx//(4*self.steps)][idx%4][idx%self.steps]
        action = self.actions[idx//(4*self.steps)][idx%4][idx%self.steps]
        query = idx % 4
        waypoints = self.waypoints[idx//(4*self.steps)][idx%4]
        waypoint_idx = idx % self.steps

        return rgb, r, t, goal, action, query, waypoints, waypoint_idx

    def get_sample(self, _idx, _action, _step):
        rgb = self.rgb_f[_idx]
        goal = self.goal[_idx][_action][_step]
        action = self.actions[_idx][_action][_step]
        waypoints = self.waypoints[_idx][_action]
        waypoint_idx = _step

        return rgb, goal, action, waypoints, waypoint_idx

if __name__ == '__main__':
    import argparse
    import cv2
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_teacher', type=Path, required=True)
    parser.add_argument('--goal_prediction', type=Path, required=True)
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--scene', type=str, required=True)
    parsed = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up source teacher
    student_args = get_model_args(parsed.source_teacher, 'student_args')
    data_args = get_model_args(parsed.source_teacher, 'data_args')
    source_teacher = GoalConditioned(**student_args, **data_args).to(device)
    source_teacher.load_state_dict(torch.load(parsed.source_teacher, map_location=device))
    source_teacher.eval()

    # set up goal prediction
    model_args = get_model_args(parsed.goal_prediction, 'model_args')
    data_args = get_model_args(parsed.goal_prediction, 'data_args')
    goal_prediction = GoalPredictionModel(**model_args).to(device)
    goal_prediction.load_state_dict(torch.load(parsed.goal_prediction, map_location=device))
    goal_prediction.eval()

    d = TargetDataset(source_teacher, goal_prediction, parsed.dataset_dir,
            parsed.scene, data_args['zoom'], data_args['steps'])

    cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('rgb', 768, 320)
    cv2.namedWindow('semantic', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('semantic', 768, 320)

    _idx, _action, _step = 0, 0, 0
    while _idx < len(d):
        rgb, goal, action, waypoints, waypoint_idx = d.get_sample(_idx, _action, _step)

        rgb = np.uint8(rgb)
        for j, (x, y) in enumerate(waypoints * d.zoom):
            _x, _y = int(10*x)+192, 80-int(10*y)
            if j == 0:
                cv2.circle(rgb, (_x, _y), 5, (0, 255, 0), -1)
            cv2.circle(rgb, (_x, _y), 4 if j == waypoint_idx else 3,
                    (255, 0, 0) if j == waypoint_idx else (0, 0, 255), -1)

        # inputs and outputs
        rgb[:25] = (255, 255, 255)
        cv2.putText(rgb, f"""Frame {_idx}, Action {ACTIONS[_action]}, \
Step {_step}          Pred {ACTIONS[action]}""",
            (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # (x, z)
        rgb[140:,:124] = (255, 255, 255)
        cv2.putText(rgb, f"""({d.zoom*d.waypoints[_idx, _action, _step, 0].item():.2f}, \
{d.zoom*d.waypoints[_idx, _action, _step, 1].item():.2f})""",
            (0, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # (r, theta)
        rgb[140:,260:] = (255, 255, 255)
        cv2.putText(rgb, f"""({d.r[_idx, _action, _step].item():.2f}, \
{d.t[_idx, _action, _step].item():.2f})""",
            (260, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow('rgb', rgb)
        cv2.imshow('semantic', 255*np.uint8(d.onehot[_idx].reshape(160, -1)))

        key = cv2.waitKey(0)
        if key == 97:    # A
            _idx = max(0, _idx-1)
        elif key == 100: # D
            _idx = min(d.rgb_f.shape[0]-1, _idx+1)
        elif key == 106: # J
            _action = max(0, _action-1)
        elif key == 108: # L
            _action = min(3, _action+1)
        elif key == 115: # S
            _step = max(0, _step-1)
        elif key == 119: # W
            _step = min(d.steps-1, _step+1)
        elif key == 113: # Q
            cv2.destroyAllWindows()
            break
