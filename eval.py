from habitat_wrapper import Rollout, rollout_episode, TASKS, MODELS, METRICS
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', '-n', type=int, default=5)
    parser.add_argument('--model', '-m', type=Path, required=True)
    parser.add_argument('--auto', '-a', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = get_model(**get_model_args(args.model, 'student_args')).to(device)
    print(device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    summary = defaultdict(float)
    summary['ep'] = 1
    if (args.model.parent / 'summary.csv').exists():
        summary = pd.read_csv(args.model.parent / 'summary.csv').iloc[0]

    teacher_args = get_model_args(args.model, 'teacher_args')
    teacher_args['dagger'] = False
    env = Rollout(**teacher_args, model=net)
    for ep in range(int(summary['ep']), int(summary['ep'])+args.num_episodes):
        lwns, j = 0, 0

        steps = rollout_episode(env)
        for i, step in enumerate(steps):
            lwns = max(lwns, j)
            if step['is_stuck']:
                j = 0
            j += 1

            cv2.imshow('rgb', step['rgb'])
            cv2.waitKey(10 if args.auto else 0)

        print(f'[!] finish ep {ep:06}')
        print(env.env.get_metrics()['collisions'])
        print(f'LWNS: {lwns}')
        for m, v in env.env.get_metrics().items():
            if m in METRICS:
                summary[m] += v

        summary['ep'] = ep+1
        print({k: v / ep for k, v in summary.items() if k in METRICS})
        print()

        pd.DataFrame([summary]).to_csv(args.model.parent / 'summary.csv', index=False)
