from habitat_wrapper import Rollout, METRICS
from habitat_dataset import HabitatDataset
from model import get_model

import argparse
from collections import defaultdict
from operator import itemgetter
import time

import torch
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from pyquaternion import Quaternion
from PIL import Image
import yaml

def get_model_args(model, key):
    return yaml.load((model.parent / 'config.yaml').read_text())[key]['value']

def get_env(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_args = get_model_args(model, 'student_args')
    net = get_model(**student_args).to(device)
    print(device)
    net.load_state_dict(torch.load(model, map_location=device))

    teacher_args = get_model_args(model, 'teacher_args')
    env = Rollout(**teacher_args, student=net)
    env.mode = 'student'

    return env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', '-n', type=int, default=5)
    parser.add_argument('--models_root', '-r', type=Path, required=True)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--epoch', '-e', type=int)
    parser.add_argument('--auto', '-a', action='store_true')
    parser.add_argument('--human', '-hr', action='store_true')
    parser.add_argument('--display', '-d', action='store_true')
    parsed = parser.parse_args()

    model_path = parsed.models_root / parsed.model
    if not parsed.epoch:
        model = [model.stem for model in model_path.glob('model_*.t7')][-1] + '.t7'
        model_path = model_path / model
    else:
        model_path = model_path / f'model_{parsed.epoch:03}.t7'

    task = get_model_args(model_path, 'teacher_args')['task']

    env = get_env(model_path)
    for ep in range(parsed.num_episodes):
        lwns, longest, length = 0, 0, 0

        for i, step in enumerate(env.rollout()):
            lwns = max(lwns, longest)
            if step['is_stuck']:
                longest = 0
                print('stuck?')
                #break
            longest += 1
            length += 1

            if parsed.display:
                cv2.imshow('rgb', step['rgb'])
                cv2.waitKey(10 if parsed.auto else 0)

        if task == 'dontcrash':
            if parsed.human:
                print(f'[!] Finish Episode {ep:06}, Length: {length}, LWNS: {lwns}, LWNS_norm: {lwns/length}\n')
            else:
                print(f'{lwns},{lwns/length}')
        elif task == 'pointgoal':
            source_position = env.state.position
            rot = env.state.rotation.components
            source_rotation = Quaternion(*rot[1:4], rot[0])
            goal_position = env.env.current_episode.goals[0].position
            direction = HabitatDataset.get_direction(source_position, source_rotation, goal_position).unsqueeze(dim=0)
            spl, success, dtg = itemgetter('spl', 'success', 'distance_to_goal')(env.env.get_metrics())

            print(f'[!] Finish Episode {ep:06}, DTG: {dtg}, Success: {success}, SPL: {spl}, Direction: {direction}\n')
