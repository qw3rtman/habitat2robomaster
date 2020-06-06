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

from wrapper import Rollout, METRICS

import sys
sys.path.append('/u/nimit/Documents/robomaster/habitat2robomaster')
from model import get_model
from util import C, make_onehot

BACKGROUND = (0,0,0,0)
COLORS = [
    (0,47,0,150),
    (253,253,17,150)
]

ACTIONS = ['S', 'F', 'L', 'R']

def get_fig(xy):
    fig = go.Figure(data=[go.Box(y=y,
        boxpoints='all',
        boxmean=True,
        jitter=0.1,
        pointpos=-1.6,
        name=f'{x}',
        marker_color=['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 20)][i%20]
    ) for i, (x, y) in enumerate(xy.items())])
    fig.update_layout(
        xaxis=dict(title='Scene', showgrid=False, zeroline=False, dtick=1),
        yaxis=dict(zeroline=False, gridcolor='white'),
        paper_bgcolor='rgb(233,233,233)',
        plot_bgcolor='rgb(233,233,233)',
        showlegend=False
    )

    return fig

all_success, all_spl, all_softspl, total = {}, {}, {}, 0
def visualize(parsed):
    global total

    print(f'[!] Start {parsed.scene}')
    env = Rollout('pointgoal', 'depth', 'semantic', mode=parsed.mode,
            shuffle=True, split=parsed.split, dataset=parsed.dataset,
            sensors=['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR'],
            scenes=parsed.scene, height=parsed.height, width=parsed.width,
            fov=parsed.fov, camera_height=parsed.camera_height)

    success = np.zeros(parsed.num_episodes)
    spl = np.zeros(parsed.num_episodes)
    softspl = np.zeros(parsed.num_episodes)

    all_success[parsed.scene] = success
    all_spl[parsed.scene] = spl
    all_softspl[parsed.scene] = softspl

    for ep in range(parsed.num_episodes):
        print(f'[!] Start {parsed.scene} ({total})')
        total += 1
        images = []

        env.clean()
        for i, step in enumerate(env.rollout()):
            if i == 0:
                dtg = env.env.get_metrics()['distance_to_goal']

            frame = Image.fromarray(step['rgb'])
            if env.target == 'semantic':
                mapping = parsed.scene if parsed.dataset == 'replica' else None
                onehot = make_onehot(step['semantic'], scene=mapping)

                semantic = np.zeros((parsed.height, parsed.width, 4), dtype=np.uint8)
                semantic[...] = BACKGROUND
                for i in range(min(onehot.shape[-1], len(COLORS))):
                    semantic[onehot[...,i] == 1] = COLORS[i]
                semantic = Image.fromarray(semantic, 'RGBA')
                frame = Image.alpha_composite(frame.convert('RGBA'), semantic)

            draw = ImageDraw.Draw(frame)
            font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf', 18)
            direction = env.get_direction()
            draw.rectangle((0, 0, 255, 20), fill='black')

            _action = ACTIONS[step['action'][env.mode]['action']]
            draw.text((0, 0), '({: <5.1f}, {: <5.1f}) {: <4.1f} {}'.format(*direction, env.env.get_metrics()['distance_to_goal'], _action), fill='white', font=font)

            images.append(np.transpose(np.uint8(frame), (2, 0, 1)))

        metrics = env.env.get_metrics()
        success[ep] = metrics['success']
        spl[ep] = metrics['spl']
        softspl[ep] = metrics['softspl']
        print(f'[!] End {parsed.scene} ({total})', metrics)

        log = {f'{parsed.scene}_video': wandb.Video(np.array(images), fps=20, format='mp4'),
                'success_mean': np.sum(np.concatenate([_success for _success in all_success.values()])) / total,
                'spl_mean': np.sum(np.concatenate([_spl for _spl in all_spl.values()])) / total,
                'softspl_mean': np.sum(np.concatenate([_softspl for _softspl in all_softspl.values()])) / total}
        wandb.run.summary['episode'] += 1
        wandb.log(
                {('%s/%s' % ('val', k)): v for k, v in log.items()},
                step=wandb.run.summary['episode'])

    env.env.close()
    del env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--scene', default='*')
    parser.add_argument('--num_episodes', default=10)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--fov', type=int, default=90)
    parser.add_argument('--camera_height', type=float, default=1.5)
    parsed = parser.parse_args()

    wandb.init(project='pointgoal-visualize')
    wandb.config.update(parsed)
    wandb.run.summary['episode'] = 0

    with torch.no_grad():
        visualize(parsed)

    log = {
        'spl': get_fig(all_spl),
        'softspl': get_fig(all_softspl)
    }
    wandb.run.summary['episode'] += 1
    wandb.log(
        {('%s/%s' % ('val', k)): v for k, v in log.items()},
        step=wandb.run.summary['episode'])
