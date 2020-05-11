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

from habitat_wrapper import Rollout, get_episode, METRICS
from habitat_dataset import HabitatDataset
from model import get_model

COLORS = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 20)]
def get_fig(xy):
    fig = go.Figure(data=[go.Box(y=y,
        boxpoints='all',
        boxmean=True,
        jitter=0.1,
        pointpos=-1.6,
        name=f'{x}',
        marker_color=COLORS[i%20]
    ) for i, (x, y) in enumerate(xy.items())])
    fig.update_layout(
        xaxis=dict(title='Scene', showgrid=False, zeroline=False, dtick=1),
        yaxis=dict(zeroline=False, gridcolor='white'),
        paper_bgcolor='rgb(233,233,233)',
        plot_bgcolor='rgb(233,233,233)',
        showlegend=False
    )

    return fig

NUM_EPISODES = 10

all_success, all_spl, all_softspl, total = {}, {}, {}, 0
def _eval_scene(scene, parsed):
    global total

    split = f'{parsed.split}' if student_args['target'] == 'semantic' else f'{parsed.split}_ddppo'
    print(split)
    sensors = ['RGB_SENSOR']
    if student_args['target'] == 'semantic':
        sensors.append('SEMANTIC_SENSOR')
    else:
        sensors.append('DEPTH_SENSOR')
    env = Rollout(task='pointgoal', proxy=student_args['target'], student=net, split=f'{split}', mode='student', rnn=student_args['rnn'], shuffle=True, dataset=data_args['scene'], sensors=sensors, scenes=scene, compass=parsed.compass)

    print(f'[!] Start {scene}')
    success = np.zeros(NUM_EPISODES)
    spl = np.zeros(NUM_EPISODES)
    softspl = np.zeros(NUM_EPISODES)

    all_success[scene] = success
    all_spl[scene] = spl
    all_softspl[scene] = softspl

    for ep in range(NUM_EPISODES):
        total += 1

        net.clean()
        images = []

        for i, step in enumerate(get_episode(env)):
            if i == 0:
                dtg = env.env.get_metrics()['distance_to_goal']

            frame = Image.fromarray(step['rgb'])
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

        #print(f'[{ep+1}/NUM_EPISODES] [{scene}] Success: {metrics["success"]}, SPL: {metrics["spl"]:.02f}, SoftSPL: {metrics["softspl"]:.02f}, DTG -> DFG: {dtg:.02f} -> {metrics["distance_to_goal"]:.02f}')

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
    parser.add_argument('--split', required=True)
    parser.add_argument('--compass', action='store_true')
    parsed = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_args = get_model_args(parsed.model, 'teacher_args')
    student_args = get_model_args(parsed.model, 'student_args')
    data_args = get_model_args(parsed.model, 'data_args')

    input_channels = 3
    if student_args['target'] == 'semantic':
        input_channels = HabitatDataset.NUM_SEMANTIC_CLASSES
    net = get_model(**student_args, tgt_mode='ddppo' if parsed.compass else 'nimit', input_channels=input_channels).to(device)
    net.load_state_dict(torch.load(parsed.model, map_location=device))
    net.batch_size=1
    net.eval()

    run_name = f"{get_model_args(parsed.model)['run_name']['value']}-{parsed.model.stem}-{parsed.split}-new"
    wandb.init(project='pointgoal-rgb2depth-eval', id=run_name, config=get_model_args(parsed.model))
    wandb.run.summary['episode'] = 0

    if student_args['target'] == 'semantic':
        with open('splits/mp3d_val.txt', 'r') as f:
            scenes = [scene.strip() for scene in f.readlines()]
        available_scenes = scenes
        print(scenes)
    else:
        #available_scenes = set([scene.stem.split('.')[0] for scene in Path('/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/train_ddppo/content').glob('*.gz')])
        available_scenes = set([scene.stem.split('.')[0] for scene in Path('/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/val_ddppo/content').glob('*.gz')])
        with open('splits/gibson_splits/train_val_test_fullplus.csv', 'r') as csv_f:
            splits = pd.read_csv(csv_f)
        #scenes = splits[splits['train'] ==1]['id'].tolist()
        scenes = splits[splits['val'] ==1]['id'].tolist()
    with torch.no_grad():
        for scene in scenes:
            if scene not in available_scenes:
                continue
            _eval_scene(scene, parsed)

    log = {
        'spl': get_fig(all_spl),
        'softspl': get_fig(all_softspl)
    }
    wandb.run.summary['episode'] += 1
    wandb.log(
        {('%s/%s' % ('val', k)): v for k, v in log.items()},
        step=wandb.run.summary['episode'])
