import wandb
import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import tqdm

from wrapper import Rollout, METRICS


all_success, all_spl, all_softspl, total = {}, {}, {}, 0
def _eval_scene(scene, parsed, num_episodes):
    global total

    split = f'{parsed.split}_ddppo'
    if parsed.dataset != 'gibson':
        split = f'{parsed.split}'
    sensors = ['DEPTH_SENSOR']
    dataset = parsed.dataset
    print(split, sensors, dataset)

    print(f'[!] Start {scene}')
    env = Rollout('pointgoal', 'depth', 'rgb', mode='teacher',
            shuffle=True, split=split, dataset=dataset,
            sensors=sensors, scenes=scene, k=parsed.k)

    success = np.zeros(num_episodes)
    spl = np.zeros(num_episodes)
    softspl = np.zeros(num_episodes)

    all_success[scene] = success
    all_spl[scene] = spl
    all_softspl[scene] = softspl

    for ep in range(num_episodes):
        total += 1

        env.clean()
        for step in env.rollout():
            pass

        metrics = env.env.get_metrics()
        success[ep] = metrics['success']
        spl[ep] = metrics['spl']
        softspl[ep] = metrics['softspl']

        print(metrics)
        log = {'success_mean': np.sum(np.concatenate([_success for _success in all_success.values()])) / total,
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
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--k', type=int, required=True)
    parsed = parser.parse_args()

    run_name = f'{parsed.dataset}-{parsed.split}-k={parsed.k}'
    wandb.init(project='pointgoal-depth2rgb-tune-dagger', id=run_name)
    wandb.run.summary['episode'] = 0

    num_episodes = 10
    if parsed.dataset in ['gibson', 'mp3d']:
        available_scenes = set([scene.stem.split('.')[0] for scene in Path(f'/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/{parsed.split}_ddppo/content').glob('*.gz')])
        with open('splits/gibson_splits/train_val_test_fullplus.csv', 'r') as csv_f:
            splits = pd.read_csv(csv_f)
        scenes = splits[splits[parsed.split] ==1]['id'].tolist()
    else: # castle, office
        scenes = ['*']
        available_scenes = set(scenes)
        num_episodes = 100

    for scene in scenes:
        if scene not in available_scenes:
            continue
        _eval_scene(scene, parsed, num_episodes)
