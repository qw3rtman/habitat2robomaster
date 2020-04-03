from habitat_wrapper import Rollout, rollout_episode, models, METRICS
from model import Network

import argparse
from collections import defaultdict
import time

import torch
import cv2
import pandas as pd
from pathlib import Path
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: take model_path and config_path
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--num_episodes', type=int, required=True)
    parser.add_argument('--input_type', choices=models.keys(), required=True)
    args = parser.parse_args()

    net = Network() # TODO: read config.yaml, pass in model_args
    net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    summary = defaultdict(float)
    summary['ep'] = 1
    if (args.model_path.parent / 'summary.csv').exists():
        summary = pd.read_csv(args.model_path.parent / 'summary.csv').iloc[0]

    env = Rollout(args.input_type, model=net)
    for ep in range(int(summary['ep']), int(summary['ep'])+args.num_episodes):
        steps = rollout_episode(env)
        for step in steps:
            #img = Image.fromarray(step['rgb'])
            #img.show()
            #time.sleep(0.25)
            cv2.imshow('rgb', step['rgb'])
            cv2.waitKey(1)

        print(f'[!] finish ep {ep:04}')
        for m, v in env.env.get_metrics().items():
            if m in METRICS:
                summary[m] += v

        summary['ep'] = ep+1
        print({k: v / ep for k, v in summary.items() if k in METRICS})
        print()

        pd.DataFrame([summary]).to_csv(args.model_path.parent / 'summary.csv', index=False)
