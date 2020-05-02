import gzip
import json
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

import argparse
from pathlib import Path
from itertools import repeat
from collections import defaultdict
import gc

import torch
import pandas as pd
import tqdm
import yaml

from habitat_wrapper import Rollout, get_episode, METRICS
from model import get_model

def _eval_scene(scene, parsed):
    env = Rollout(**teacher_args, student=net, split='val_ddppo', mode='student', rnn=student_args['rnn'], shuffle=True, dataset=data_args['scene'], sensors=['RGB_SENSOR', 'DEPTH_SENSOR'], scenes=scene)

    summary = defaultdict(float)
    print(f'[!] Start {scene}')
    for ep in range(10):
        for i, step in enumerate(get_episode(env)):
            if i == 0:
                dtg = env.env.get_metrics()['distance_to_goal']
            #print(f'[{i:03}] {scene}')

        metrics = env.env.get_metrics()
        for m, v in metrics.items():
            if m in METRICS:
                summary[m] += v

        print(f'[{ep+1}/10] [{scene}] Success: {metrics["success"]}, SPL: {metrics["spl"]:.02f}, SoftSPL: {metrics["softspl"]:.02f}, DTG -> DFG: {dtg:.02f} -> {metrics["distance_to_goal"]:.02f}')

    pd.DataFrame([summary]).to_csv(Path('/u/nimit/Documents/robomaster/habitat2robomaster/eval_results') / f'{scene}_summary.csv', index=False)

    """
    env.env.close()
    del env
    del net
    gc.collect()
    torch.cuda.empty_cache()
    """

def get_model_args(model, key):
    return yaml.load((model.parent / 'config.yaml').read_text())[key]['value']

if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)   # input
    #parser.add_argument('--scene', required=True)
    parsed = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_args = get_model_args(parsed.model, 'teacher_args')
    student_args = get_model_args(parsed.model, 'student_args')
    data_args = get_model_args(parsed.model, 'data_args')

    net = get_model(**student_args).to(device)
    net.load_state_dict(torch.load(parsed.model, map_location=device))
    net.batch_size=1
    net.eval()

    scenes = [scene.stem.split('.')[0] for scene in Path('/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/val_ddppo/content').glob('*.gz')]
    with torch.no_grad():
        for scene in scenes:
            _eval_scene(scene, parsed)

    """
    with torch.no_grad():
        with mp.Pool(2) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
            #for _ in pool.imap_unordered(_eval_scene, scenes):
            for out in pool.starmap(_eval_scene, zip(scenes, repeat(parsed))):
                out.get()
                pbar.update()
    """
