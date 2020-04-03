import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

from pathlib import Path
import argparse
import time
from collections import defaultdict

import habitat
from habitat.config import Config
from habitat.config.default import get_config
from habitat.core.env import Env
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.agents.ppo_agents import PPOAgent
from gym.spaces import Box, Dict, Discrete

models = {
    'rgb':   '/scratch/cluster/nimit/models/habitat/ppo/rgb.pth',
    'depth':   '/scratch/cluster/nimit/models/habitat/ppo/depth.pth',
    #'rgb':   '/Users/nimit/Documents/robomaster/habitat/models/v2/rgb.pth',
    #'depth': '/Users/nimit/Documents/robomaster/habitat/models/v2/depth.pth'
}

configs = {
    'rgb':   'rgb_test.yaml',
    'depth': 'depth_test.yaml'
}

parser = argparse.ArgumentParser()
parser.add_argument('--input_type', choices=models.keys(), required=True)
parser.add_argument('--num_episodes', type=int, required=True)
parser.add_argument('--dataset_dir', type=Path, required=True)
parsed = parser.parse_args()

# SETUP ########################################################################
c = Config()

c.RESOLUTION       = 256
c.HIDDEN_SIZE      = 512
c.RANDOM_SEED      = 7

c.INPUT_TYPE       = parsed.input_type
c.MODEL_PATH       = models[c.INPUT_TYPE]
c.GOAL_SENSOR_UUID = 'pointgoal_with_gps_compass'

c.freeze()

env = Env(config=get_config(configs[c.INPUT_TYPE]))
agent = PPOAgent(c)
################################################################################

agg_metrics: Dict = defaultdict(float)

ep = 1
while ep <= parsed.num_episodes:
    episode_dir = parsed.dataset_dir / f'{ep:03}'
    episode_dir.mkdir(parents=True, exist_ok=True)

    agent.reset()
    observations = env.reset()

    i = 0
    stats = list()
    while not env.episode_over: # TODO: stop if no movement in 50, prune bad/stuck episodes as in supertux PPO
        action = agent.act(observations)  # t
        state = env.sim.get_agent_state() # t

        position = state.position
        rotation = state.rotation.components
        stats.append({
            'step': i,
            'action': action['action'],
            'x': position[0],
            'y': position[1],
            'z': position[2],
            'rot_i': rotation[0],
            'rot_j': rotation[1],
            'rot_k': rotation[2],
            'rot_l': rotation[3]
        })

        Image.fromarray(observations['rgb']).save(episode_dir / f'rgb_{i:04}.png')
        np.save(episode_dir / f'sem_{i:04}', observations['semantic'])

        observations = env.step(action)
        i += 1
    pd.DataFrame(stats).to_csv(episode_dir / 'episode.csv', index=False)

    metrics = env.get_metrics()
    pd.DataFrame([metrics]).to_csv(episode_dir / 'metrics.csv', index=False)

    for m, v in metrics.items():
        agg_metrics[m] += v

    print(f'[!] finish ep {ep:03}')
    print({k: v / ep for k, v in agg_metrics.items()})
    print()

    ep += 1
