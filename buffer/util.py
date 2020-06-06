import torch
from pathlib import Path
import numpy as np
import json

C = 1
def make_onehot(semantic, scene=None):
    """
        input:  torch (B,H,W,1), dtype: torch/np.uint8
        output: torch (B,H,W,C), dtype: torch.float
    """
    onehot = torch.zeros((*semantic.shape, C), dtype=torch.float)
    if scene is not None: # replica mapping
        mapping_f = Path(f'/scratch/cluster/nimit/habitat/habitat-api/data/scene_datasets/replica/{scene}/habitat/info_semantic.json')
        with open(mapping_f) as f:
            j = json.load(f)
        instance_to_class = np.array(j['id_to_label'])
        class_names = {_class['name']: _class['id'] for _class in j['classes']}
        classes = instance_to_class[semantic]
        floor = np.array([class_names['floor'], class_names['rug'], class_names['stair'],
                 class_names['shower-stall'], class_names['basket']])
        onehot[..., 0] = torch.as_tensor(np.isin(classes, floor), dtype=torch.float)
        #onehot[..., 1] = torch.as_tensor(classes == class_names['wall'], dtype=torch.float)
    else: # handle in env.py#step/#reset; TODO: move that logic to here
        onehot[..., 0] = torch.as_tensor(semantic==2, dtype=torch.float)
        #onehot[..., 1] = torch.as_tensor((semantic!=2)&(semantic!=17)&(semantic!=28), dtype=torch.float)
    return onehot
