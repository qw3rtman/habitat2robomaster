from habitat_wrapper import Rollout, rollout_episode, models, METRICS
from model import DirectImitation, ConditionalImitation

import argparse
from collections import defaultdict
import time

import torch
import cv2
import pandas as pd
from pathlib import Path
from PIL import Image

NETWORKS = ['direct', 'conditional']
def _get_network(network):
    if network == 'direct':
        return DirectImitation()
    elif network == 'conditional':
        return ConditionalImitation()

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
        steps = rollout_episode(env)
        for i, step in enumerate(steps):
            #img = Image.fromarray(step['rgb'])
            #img.show()
            #time.sleep(0.25)
            print(i)
            print()
            cv2.imshow('rgb', step['rgb'])
            cv2.waitKey(10 if args.auto else 0)

        print(f'[!] finish ep {ep:04}')
        for m, v in env.env.get_metrics().items():
            if m in METRICS:
                summary[m] += v

        summary['ep'] = ep+1
        print({k: v / ep for k, v in summary.items() if k in METRICS})
        print()

        pd.DataFrame([summary]).to_csv(args.model_path.parent / 'summary.csv', index=False)
