from habitat_wrapper import get_rollout, METRICS
from model import get_model

import argparse
from collections import defaultdict
import time

import torch
import cv2
import pandas as pd
from pathlib import Path
from PIL import Image
import yaml

def get_model_args(model, key):
    return yaml.load((model.parent / 'config.yaml').read_text())[key]['value']

def get_env(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = get_model(**get_model_args(model, 'student_args')).to(device)
    print(device)
    net.load_state_dict(torch.load(model, map_location=device))

    teacher_args = get_model_args(model, 'teacher_args')
    env = get_rollout(**teacher_args, student=net)
    env.mode = 'student'

    return env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', '-n', type=int, default=5)
    parser.add_argument('--models_root', '-r', type=Path, required=True)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--epoch', '-e', type=int)
    parser.add_argument('--auto', '-a', action='store_true')
    parsed = parser.parse_args()

    model_path = parsed.models_root / parsed.model
    if not parsed.epoch:
        model = [model.stem for model in model_path.glob('model_*.t7')][-1] + '.t7'
        model_path = model_path / model
    else:
        model_path = model_path / f'model_{parsed.epoch:03}.t7'

    summary = defaultdict(float)
    summary['ep'] = 1
    if (model_path.parent / 'summary.csv').exists():
        summary = pd.read_csv(model_path.parent / 'summary.csv').iloc[0]

    env = get_env(model_path)
    for ep in range(int(summary['ep']), int(summary['ep'])+parsed.num_episodes):
        lwns, longest, length = 0, 0, 0

        for i, step in enumerate(env.rollout()):
            lwns = max(lwns, longest)
            if step['is_stuck']:
                longest = 0
            longest += 1
            length += 1

            cv2.imshow('rgb', step['rgb'])
            cv2.waitKey(10 if parsed.auto else 0)

        print(f'[!] Finish Episode {ep:06}, LWNS: {lwns}, LWNS_norm: {lwns/length}\n')
        """
        print(env.env.get_metrics()['collisions'])
        for m, v in env.env.get_metrics().items():
            if m in METRICS:
                summary[m] += v

        summary['ep'] = ep+1
        print({k: v / ep for k, v in summary.items() if k in METRICS})
        print()

        pd.DataFrame([summary]).to_csv(model_path.parent / 'summary.csv', index=False)
        """
