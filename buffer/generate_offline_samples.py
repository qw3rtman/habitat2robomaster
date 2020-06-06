import gzip
import json

import wandb
import argparse
from pathlib import Path
from itertools import repeat
from collections import defaultdict
import gc

import torch
import pandas as pd
import numpy as np
import tqdm
import yaml
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

from wrapper import Rollout, save_episode

import sys
sys.path.append('/u/nimit/Documents/robomaster/habitat2robomaster')
from model import get_model
from util import C, make_onehot


def generate_samples(parsed):
    print(f'[!] Start {parsed.scene}')
    env = Rollout('pointgoal', 'depth', parsed.target, mode=parsed.mode,
            shuffle=True, split=parsed.split, dataset=parsed.dataset,
            sensors=[f'{parsed.target.upper()}_SENSOR', 'DEPTH_SENSOR'],
            scenes=parsed.scene, height=parsed.height, width=parsed.width,
            fov=parsed.fov, camera_height=parsed.camera_height)

    success, spl, softspl = [], [], []
    for ep in range(parsed.num_episodes):
        print(f'[!] Start {parsed.scene} ({ep})')
        save_episode(env, parsed.dataset_dir / f'{ep:06}')

        metrics = env.env.get_metrics()
        success.append(metrics['success'])
        spl.append(metrics['spl'])
        softspl.append(metrics['softspl'])
        print(f'[!] End {parsed.scene} ({ep})', metrics)

        wandb.run.summary['episode'] += 1
        wandb.run.summary['success'] = np.mean(success)
        wandb.run.summary['spl'] = np.mean(spl)
        wandb.run.summary['softspl'] = np.mean(softspl)

    env.env.close()
    del env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--target', choices=['rgb', 'semantic'], required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--scene', default='*')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--fov', type=int, default=90)
    parser.add_argument('--camera_height', type=float, default=1.5)
    parsed = parser.parse_args()

    wandb.init(project='pointgoal-generate-offline-samples')
    wandb.config.update(parsed)
    wandb.run.summary['episode'] = 0

    generate_samples(parsed)
