from pathlib import Path
import yaml
import os
import argparse
import shutil

import wandb
import torch
import pandas as pd
import numpy as np

from habitat_wrapper import Rollout, get_episode, save_episode
from habitat_dataset import HabitatDataset
from model import get_model
import random

BUFFER_CAPACITY, NUM_SCENES, NUM_EPISODES = 768, 32, 8
success, spl, softspl = [], [], []
def collect(scene, parsed):
    split = 'train' if (student_args['target'] == 'semantic' or datasets[scene] == 'mp3d') else 'train_ddppo'
    print(split, datasets[scene])
    sensors = ['DEPTH_SENSOR']
    if student_args['target'] == 'semantic':
        sensors.append('SEMANTIC_SENSOR')
    elif student_args['target'] == 'rgb':
        sensors.append('RGB_SENSOR')
    env = Rollout(task='pointgoal', save=student_args['target'], proxy=teacher_args['proxy'], target=student_args['target'], student=net, split=split, mode='both', rnn=student_args['rnn'], shuffle=True, dataset=datasets[scene], sensors=sensors, scenes=scene)
    env.epoch = parsed.epoch # for beta term in DAgger

    print(f'[!] Start {scene}')
    for ep in range(ep_start, ep_start+NUM_EPISODES):
        split_dir = dagger_dir/'train' if np.random.random() < 0.95 else dagger_dir/'val'
        episode_dir = split_dir/f'{scene}-{ep:06}'
        save_episode(env, episode_dir, max_len=200)

        metrics = env.env.get_metrics()
        success.append(metrics['success'])
        spl.append(metrics['spl'])
        softspl.append(metrics['softspl'])

        log = {
            'success_mean': np.mean(success),
            'success_median': np.median(success),
            'spl_mean': np.mean(spl),
            'spl_median': np.median(spl),
            'softspl_mean': np.mean(softspl),
            'softspl_median': np.median(softspl)
        }

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
    parser.add_argument('--dagger_dir', type=Path)
    parsed = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_args = get_model_args(parsed.model, 'teacher_args')
    student_args = get_model_args(parsed.model, 'student_args')
    data_args = get_model_args(parsed.model, 'data_args')

    dagger_dir = Path(parsed.dagger_dir) if parsed.dagger_dir is not None else Path(data_args['dagger_dir'])
    dagger_dir.mkdir(parents=True, exist_ok=True)
    (dagger_dir/'train').mkdir(parents=True, exist_ok=True)
    (dagger_dir/'val').mkdir(parents=True, exist_ok=True)

    input_channels = 3
    if student_args['target'] == 'depth':
        input_channels = 1
    elif student_args['target'] == 'semantic':
        input_channels = HabitatDataset.NUM_SEMANTIC_CLASSES
    net = get_model(**student_args, tgt_mode='nimit', input_channels=input_channels).to(device)
    net.load_state_dict(torch.load(parsed.model, map_location=device))
    net.batch_size=1
    net.eval()

    config = get_model_args(parsed.model)
    config['epoch'] = parsed.epoch
    run_name = f"{config['run_name']['value']}-epoch_{parsed.epoch:04}"
    wandb.init(project='pointgoal-rgb2depth-eval', id=run_name, config=config)
    wandb.run.summary['episode'] = 0

    # count number of directories, if over 1000, delete the NUM_SCENES*NUM_EPISODES oldest ones
    episodes = list((dagger_dir/'train').iterdir()) #+ list((dagger_dir/'val').iterdir())

    ep_start = 0
    if len(episodes) > 0:
        ep_start = max([int(episode.stem.split('-')[-1]) for episode in episodes]) + 1
    print(ep_start)
    if len(episodes) > BUFFER_CAPACITY:
        episodes.sort(key=os.path.getmtime)

        num_prune = len(episodes) - BUFFER_CAPACITY
        prunable_episodes = episodes[:-NUM_SCENES*NUM_EPISODES] # keep the last NUM_SCENES*NUM_EPISODES episodes
        prune = random.sample(prunable_episodes, num_prune)

        for episode_dir in prune:
            shutil.rmtree(episode_dir, ignore_errors=True)

    datasets = {}
    available_scenes = [scene.stem.split('.')[0] for scene in Path('/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/train_ddppo/content').glob('*.gz')] + [scene.stem.split('.')[0] for scene in Path('/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/mp3d/v1/train/content').glob('*.gz')]

    with open('splits/mp3d_train.txt', 'r') as f:
        mp3d_scenes = [scene.strip() for scene in f.readlines()]
    for scene in mp3d_scenes:
        datasets[scene] = 'mp3d'

    scenes = mp3d_scenes
    if student_args['target'] in ['rgb', 'depth']: # + Gibson
        with open('splits/gibson_splits/train_val_test_fullplus.csv', 'r') as csv_f:
            splits = pd.read_csv(csv_f)
        gibson_scenes = splits[splits['train'] ==1]['id'].tolist()
        for scene in gibson_scenes:
            datasets[scene] = 'gibson'

        scenes += gibson_scenes

    random.shuffle(scenes)
    print(scenes)
    with torch.no_grad():
        i = 0
        while i < len(scenes):
            scene = scenes[i]
            if i >= NUM_SCENES:
                break

            i += 1
            if scene not in available_scenes:
                continue
            collect(scene, parsed)
