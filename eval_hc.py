import gzip
import json
import multiprocessing
import argparse
from pathlib import Path
from operator import itemgetter
from collections import defaultdict

import torch
import pandas as pd
import tqdm
import yaml

from habitat_wrapper import Rollout, get_episode, METRICS
from model import get_model

NUM_EPISODES = 10

def _generate_fn(scene):
    env = Rollout(**teacher_args, student=net, split='val_ddppo', mode='student', rnn=student_args['rnn'], shuffle=True, dataset=data_args['scene'], sensors=['RGB_SENSOR', 'DEPTH_SENSOR'], scenes=[scene])

    summary = defaultdict(float)
    print(f'[!] Start {scene}')
    for ep in range(10):
        print(f'[{ep}/10] {scene}')
        for i, step in enumerate(get_episode(env)):
            pass

        for m, v in env.env.get_metrics().items():
            if m in METRICS:
                summary[m] += v

    pd.DataFrame([summary]).to_csv(Path('/u/nimit/Documents/robomaster/habitat2robomaster/eval_results') / f'{scene}_summary.csv', index=False)

    del env


def get_model_args(model, key):
    return yaml.load((model.parent / 'config.yaml').read_text())[key]['value']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)   # input
    parser.add_argument('--scene', required=True)
    parsed = parser.parse_args()

    teacher_args = get_model_args(parsed.model, 'teacher_args')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_args = get_model_args(parsed.model, 'student_args')
    data_args = get_model_args(parsed.model, 'data_args')
    net = get_model(**student_args).to(device)
    net.load_state_dict(torch.load(parsed.model, map_location=device))
    net.batch_size=1
    net.eval()

    #scenes = [scene.stem.split('.')[0] for scene in Path('/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/val_ddppo/content').glob('*.gz')]
    with torch.no_grad():
        _generate_fn(parsed.scene)
    #for scene in scenes:

    """
    with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        for _ in pool.imap_unordered(_generate_fn, scenes):
            pbar.update()
    """
