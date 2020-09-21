import wandb
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .wrapper import Rollout, save_episode


def generate_samples(parsed):
    print(f'[!] Start {parsed.scene}')
    env = Rollout(shuffle=True, split=parsed.split, dataset=parsed.dataset, scenes=parsed.scene)

    success, spl, softspl = [], [], []
    for ep in range(parsed.num_episodes):
        print(f'[!] Start {parsed.scene} ({ep})')
        save_episode(env, parsed.dataset_dir / f'{ep:06}')

        metrics = env.env.get_metrics()
        success.append(metrics['success'])
        spl.append(metrics['spl'])
        softspl.append(metrics['softspl'])
        print(f'[!] End {parsed.scene} ({ep})', metrics)

        wandb.run.summary['frames'] += env.i
        wandb.run.summary['episode'] += 1
        wandb.run.summary['success'] = np.mean(success)
        wandb.run.summary['spl'] = np.mean(spl)
        wandb.run.summary['softspl'] = np.mean(softspl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--scene', default='*')
    parser.add_argument('--num_episodes', type=int, default=10)
    parsed = parser.parse_args()

    wandb.init(project='pointgoal-generate-offline-samples')
    wandb.config.update(parsed)
    wandb.run.summary['frames'] = 0
    wandb.run.summary['episode'] = 0

    generate_samples(parsed)
