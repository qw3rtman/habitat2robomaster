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
from util import C, make_onehot, get_model_args

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
def _eval_scene(scene, parsed, num_episodes):
    global total

    split = f'{parsed.split}_ddppo'
    if student_args['target'] == 'semantic' or dataset != 'gibson':
        split = f'{parsed.split}'
    print(split)
    sensors = ['RGB_SENSOR']
    if student_args['target'] == 'semantic':
        sensors.append('SEMANTIC_SENSOR')
    elif student_args['target'] == 'depth':
        sensors.append('DEPTH_SENSOR')
    print(dataset)

    print(f'[!] Start {scene}')
    env = Rollout('pointgoal', teacher_args['proxy'],
            student_args['target'], mode='student', shuffle=True,
            split=split, dataset=dataset, student=net,
            sensors=sensors, scenes=scene, goal=parsed.goal,
            **data_args)

    success = np.zeros(num_episodes)
    spl = np.zeros(num_episodes)
    softspl = np.zeros(num_episodes)

    all_success[scene] = success
    all_spl[scene] = spl
    all_softspl[scene] = softspl

    for ep in range(num_episodes):
        print(f'[!] Start {scene} ({total})')
        total += 1

        """
        if student_args['method']!='feedforward':
            net.clean()
        """
        images = []

        env.clean()
        for i, step in enumerate(env.rollout()):
            #print(step['compass_r'], step['compass_t'])
            if i == 0:
                dtg = env.env.get_metrics()['distance_to_goal']

            frame = Image.fromarray(step['rgb'])
            if env.target == 'semantic':
                _scene = scene if dataset == 'replica' else None
                onehot = make_onehot(step['semantic'], scene=_scene)
                semantic = np.zeros((data_args['height'], data_args['width'], 4), dtype=np.uint8)
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
        print(f'[!] End {scene} ({total})', metrics)

        #print(f'[{ep+1}/num_episodes] [{scene}] Success: {metrics["success"]}, SPL: {metrics["spl"]:.02f}, SoftSPL: {metrics["softspl"]:.02f}, DTG -> DFG: {dtg:.02f} -> {metrics["distance_to_goal"]:.02f}')

        print(total)
        log = {f'{scene}_video': wandb.Video(np.array(images), fps=20, format='mp4'),
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
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--dataset') # override's config.yaml
    parser.add_argument('--scene')   # ^^^
    parser.add_argument('--split', required=True)
    parser.add_argument('--goal', default='polar')#, choices=['polar', 'cartesian'])
    parser.add_argument('--redo', action='store_true')
    parsed = parser.parse_args()

    run_name = f"{get_model_args(parsed.model)['run_name']['value']}-model_{parsed.epoch:03}-{parsed.split}"
    exists = False
    if not parsed.redo:
        try:
            api = wandb.Api()
            run = api.run(f'qw3rtman/pointgoal-rgb2depth-eval-hc/{run_name}')
            exists = True
        except:
            pass

    if exists:
        print('already evaluated this model; check wandb')
        raise SystemExit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        teacher_args = get_model_args(parsed.model, 'teacher_args')
    except:
        teacher_args = {'proxy': 'semantic', 'target': 'rgb', 'dataset': 'replica'}
    student_args = get_model_args(parsed.model, 'student_args')
    data_args = get_model_args(parsed.model, 'data_args')
    data_args['fov'] = 120
    data_args['camera_height'] = 0.25

    net = get_model(**student_args, **data_args).to(device)
    net.load_state_dict(torch.load(parsed.model, map_location=device))
    net.batch_size=1
    net.eval()

    config = get_model_args(parsed.model)
    config['epoch'] = parsed.epoch
    config['split'] = parsed.split
    wandb.init(project='pointgoal-{}2{}-eval'.format(teacher_args['proxy'], student_args['target']), name=run_name, config=config)
    wandb.run.summary['episode'] = 0

    dataset = teacher_args['dataset']
    if parsed.dataset:
        dataset = parsed.dataset

    num_episodes = 10
    if dataset in ['gibson', 'mp3d'] and not parsed.scene:
        if student_args['target'] == 'semantic':
            with open(f'splits/mp3d_{parsed.split}.txt', 'r') as f:
                scenes = [scene.strip() for scene in f.readlines()]
            available_scenes = scenes
            print(scenes)
        else:
            available_scenes = set([scene.stem.split('.')[0] for scene in Path(f'/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/{parsed.split}_ddppo/content').glob('*.gz')])
            with open('splits/gibson_splits/train_val_test_fullplus.csv', 'r') as csv_f:
                splits = pd.read_csv(csv_f)
            #scenes = splits[splits['train'] ==1]['id'].tolist()
            scenes = splits[splits[parsed.split] ==1]['id'].tolist()
    else: # castle, office
        scenes = [parsed.scene] if parsed.scene else ['*']
        available_scenes = set(scenes)
        num_episodes = 100

    with torch.no_grad():
        for scene in scenes:
            if scene not in available_scenes:
                continue
            _eval_scene(scene, parsed, num_episodes)

    log = {
        'spl': get_fig(all_spl),
        'softspl': get_fig(all_softspl)
    }
    wandb.run.summary['episode'] += 1
    wandb.log(
        {('%s/%s' % ('val', k)): v for k, v in log.items()},
        step=wandb.run.summary['episode'])
