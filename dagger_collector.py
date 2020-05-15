from pathlib import Path
import yaml
import argparse

import wandb
import torch
import pandas as pd
import numpy as np

from habitat_wrapper import Rollout, get_episode, save_episode
from habitat_dataset import HabitatDataset
from model import get_model
import random

BUFFER_CAPACITY, NUM_SCENES, NUM_EPISODES = 1500, 20, 25
success, spl, softspl = [], [], []
def collect(scene, parsed):
    split = f'{parsed.split}' if student_args['target'] == 'semantic' else f'{parsed.split}_ddppo'
    sensors.append('DEPTH_SENSOR')
    if student_args['target'] == 'semantic':
        sensors.append('SEMANTIC_SENSOR')
    elif student_args['target'] == 'rgb':
        sensors = ['RGB_SENSOR']
    env = Rollout(task='pointgoal', proxy=student_args['target'], student=net, split=f'{split}', mode='student', rnn=student_args['rnn'], shuffle=True, dataset=data_args['scene'], sensors=sensors, scenes=scene, compass=parsed.compass)

    print(f'[!] Start {scene}')
    for ep in range(NUM_EPISODES):
        save_episode(env, data_args['dagger_dir'])

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
    parsed = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_args = get_model_args(parsed.model, 'teacher_args')
    student_args = get_model_args(parsed.model, 'student_args')
    data_args = get_model_args(parsed.model, 'data_args')
    data_args['dagger_dir'].mkdir(parents=True, exist_ok=True)

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

    if student_args['target'] == 'semantic':
        with open('splits/mp3d_train.txt', 'r') as f:
            scenes = [scene.strip() for scene in f.readlines()]
        available_scenes = scenes
    else:
        available_scenes = [scene.stem.split('.')[0] for scene in Path('/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/train_ddppo/content').glob('*.gz')]
        with open('splits/gibson_splits/train_val_test_fullplus.csv', 'r') as csv_f:
            splits = pd.read_csv(csv_f)
        scenes = splits[splits['train'] ==1]['id'].tolist()

    random.shuffle(available_scenes)
    available_scenes = set(available_scenes[:NUM_SCENES])

    # count number of directories, if over 1000, delete the NUM_SCENES*NUM_EPISODES oldest ones
    episodes = list(data_args['dagger_dir'].iterdir())
    if len(episodes) > BUFFER_CAPACITY:
        episodes.sort(key=os.path.getmtime)

        num_prune = len(episodes) - BUFFER_CAPACITY
        prunable_episodes = episodes[:-NUM_SCENES*NUM_EPISODES] # keep the last NUM_SCENES*NUM_EPISODES episodes
        prune = random.sample(prunable_episodes, num_prune)

        for episode_dir in prune:
            shutil.rmtree(episode_dir)

    with torch.no_grad():
        for scene in scenes:
            if scene not in available_scenes:
                continue
            collect(scene, parsed)
