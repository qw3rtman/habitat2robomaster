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
    if student_args['target'] == 'semantic' or teacher_args['dataset'] != 'gibson':
        split = f'{parsed.split}'
    print(split)
    sensors = ['RGB_SENSOR']
    dataset=teacher_args['dataset']
    if student_args['target'] == 'semantic':
        sensors.append('SEMANTIC_SENSOR')
    elif student_args['target'] == 'depth':
        sensors.append('DEPTH_SENSOR')
    print(dataset)

    print(f'[!] Start {scene}')
    env = Rollout('pointgoal', teacher_args['proxy'],
            student_args['target'], mode='student', shuffle=True,
            split=split, dataset=dataset, student=net,
            rnn=student_args['method']!='feedforward',
            sensors=sensors, scenes=scene, goal=parsed.goal)

    success = np.zeros(num_episodes)
    spl = np.zeros(num_episodes)
    softspl = np.zeros(num_episodes)

    all_success[scene] = success
    all_spl[scene] = spl
    all_softspl[scene] = softspl

    for ep in range(num_episodes):
        total += 1

        if student_args['method']!='feedforward':
            net.clean()
        images = []

        env.clean()
        for i, step in enumerate(env.rollout()):
            if i == 0:
                dtg = env.env.get_metrics()['distance_to_goal']

            frame = Image.fromarray(step['rgb'])
            if env.target == 'semantic':
                onehot = make_onehot(step['semantic'])
                semantic = np.zeros((256, 256, 4), dtype=np.uint8)
                semantic[...] = BACKGROUND
                for i in range(min(onehot.shape[-1], len(COLORS))):
                    semantic[onehot[...,i] == 1] = COLORS[i]
                semantic = Image.fromarray(semantic, 'RGBA')
                frame = Image.alpha_composite(frame.convert('RGBA'), semantic)

            draw = ImageDraw.Draw(frame)
            font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf', 18)
            direction = env.get_direction()
            draw.rectangle((0, 0, 255, 20), fill='black')
            draw.text((0, 0), '({: <5.1f}, {: <5.1f}) {: <4.1f}'.format(*direction, np.linalg.norm(direction)), fill='white', font=font)

            images.append(np.transpose(np.uint8(frame), (2, 0, 1)))

        metrics = env.env.get_metrics()
        success[ep] = metrics['success']
        spl[ep] = metrics['spl']
        softspl[ep] = metrics['softspl']

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

def get_model_args(model, key=None):
    config = yaml.load((model.parent / 'config.yaml').read_text())
    if not key:
        return config

    return config[key]['value']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--goal')#, choices=['polar', 'cartesian'])
    parser.add_argument('--redo', action='store_true')
    parsed = parser.parse_args()

    run_name = f"{get_model_args(parsed.model)['run_name']['value']}-model_{parsed.epoch:03}-{parsed.split}-new"
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
    teacher_args = get_model_args(parsed.model, 'teacher_args')
    student_args = get_model_args(parsed.model, 'student_args')
    data_args = get_model_args(parsed.model, 'data_args')

    input_channels = 3
    if student_args['target'] == 'depth':
        input_channels = 1
    elif student_args['target'] == 'semantic':
        input_channels = C

    #goal_size = student_args.get('goal_size', 3 if parsed.goal == 'polar' else 2)
    #hidden_size = student_args.get('hidden_size', 1024)
    net = get_model(**student_args).to(device)# goal_size=goal_size, hidden_size=hidden_size).to(device)
    net.load_state_dict(torch.load(parsed.model, map_location=device))
    net.batch_size=1
    net.eval()

    config = get_model_args(parsed.model)
    config['epoch'] = parsed.epoch
    config['split'] = parsed.split
    wandb.init(project='pointgoal-rgb2depth-eval-hc', id=run_name, config=config)
    wandb.run.summary['episode'] = 0

    num_episodes = 10
    if teacher_args['dataset'] in ['gibson', 'mp3d']:
        if student_args['target'] == 'semantic':
            with open(f'splits/mp3d_{parsed.split}.txt', 'r') as f:
                scenes = [scene.strip() for scene in f.readlines()]
            available_scenes = scenes
            print(scenes)
        else:
            #available_scenes = set([scene.stem.split('.')[0] for scene in Path('/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/train_ddppo/content').glob('*.gz')])
            available_scenes = set([scene.stem.split('.')[0] for scene in Path(f'/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/{parsed.split}_ddppo/content').glob('*.gz')])
            with open('splits/gibson_splits/train_val_test_fullplus.csv', 'r') as csv_f:
                splits = pd.read_csv(csv_f)
            #scenes = splits[splits['train'] ==1]['id'].tolist()
            scenes = splits[splits[parsed.split] ==1]['id'].tolist()
    else: # castle, office
        scenes = ['*']
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
