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

import sys
sys.path.append('/u/nimit/Documents/robomaster/habitat2robomaster')
from buffer.util import C, make_onehot
from buffer.goal_prediction import GoalPredictionModel
from buffer.target_dataset import TargetDataset, get_dataset
from model import GoalConditioned

def get_model_args(model, key=None):
    config = yaml.load((model.parent / 'config.yaml').read_text())
    if not key:
        return config

    return config[key]['value']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_teacher', type=Path, required=True)
    parser.add_argument('--goal_prediction', type=Path, required=True)
    parsed = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up source teacher
    student_args = get_model_args(parsed.student_teacher, 'student_args')
    data_args = get_model_args(parsed.student_teacher, 'data_args')
    source_teacher = GoalConditioned(**student_args, **data_args).to(device)
    source_teacher.load_state_dict(torch.load(parsed.student_teacher, map_location=device))
    source_teacher.eval()

    # set up goal prediction
    model_args = get_model_args(parsed.goal_prediction, 'model_args')
    goal_prediction = GoalPredictionModel(**model_args).to(device)
    goal_prediction.load_state_dict(torch.load(parsed.goal_prediction, map_location=device))
    goal_prediction.eval()

    d = TargetDataset(
            source_teacher,
            goal_prediction,
            Path('/scratch/cluster/nimit/data/habitat/replica-apartment_2/train/000040'),
            'apartment_2')
