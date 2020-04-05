from habitat_wrapper import Rollout, rollout_episode, models, METRICS
from model import *

import argparse
from collections import defaultdict
import time

import torch
import cv2
import pandas as pd
from pathlib import Path
from PIL import Image

NETWORKS = ['ppo-direct', 'ppo-conditional', 'ddppo-direct']
def _get_network(network):
    if network == 'ppo-direct':         # v2.x
        return DirectImitation()
    elif network == 'ppo-conditional':  # v3.x
        return ConditionalImitation()
    elif network == 'ddppo-direct': # v4.x
        return DirectImitationDDPPO()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: take model_path and config_path
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--num_episodes', type=int, required=True)
    parser.add_argument('--input_type', choices=models.keys(), required=True)
    parser.add_argument('--network', choices=NETWORKS, required=True)
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = _get_network(args.network).to(device) # TODO: read config.yaml, pass in model_args
    print(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))

    summary = defaultdict(float)
    summary['ep'] = 1
    if (args.model_path.parent / 'summary.csv').exists():
        summary = pd.read_csv(args.model_path.parent / 'summary.csv').iloc[0]

    env = Rollout(args.input_type, evaluate=True, model=net)
    for ep in range(int(summary['ep']), int(summary['ep'])+args.num_episodes):
        longest_no_stuck = 0
        j = 0

        steps = rollout_episode(env)
        for i, step in enumerate(steps):
            #print(i)
            #print()
            if step['is_stuck']:
                longest_no_stuck = max(longest_no_stuck, j)
                j = 0
            j += 1

            cv2.imshow('rgb', step['rgb'])
            cv2.waitKey(10 if args.auto else 0)

        print(f'[!] finish ep {ep:04}')
        print(env.env.get_metrics()['collisions'])
        print('longest with no stucks: {}'.format(longest_no_stuck))
        for m, v in env.env.get_metrics().items():
            if m in METRICS:
                summary[m] += v

        summary['ep'] = ep+1
        print({k: v / ep for k, v in summary.items() if k in METRICS})
        print()

        pd.DataFrame([summary]).to_csv(args.model_path.parent / 'summary.csv', index=False)
