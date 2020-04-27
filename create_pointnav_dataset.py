import gzip
import json
import multiprocessing

import argparse
from pathlib import Path
import pandas as pd

import tqdm

import habitat
import habitat_sim
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode

NUM_EPISODES = int(5e3)
GIBSON_QUALITY = Path('/u/nimit/Documents/robomaster/habitat2robomaster/splits/gibson_quality_ratings.csv')
GIBSON_SPLITS = Path('/u/nimit/Documents/robomaster/habitat2robomaster/splits/gibson_splits/train_val_test_fullplus.csv')
EPISODES_DIR = Path('/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1')
SPLIT = ''

def _generate_fn(scene):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = str(scene)
    cfg.SIMULATOR.AGENT_0.SENSORS = []
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim, NUM_EPISODES, is_gen_shortest_path=False
        )
    )
    for ep in dset.episodes:
        #ep.scene_id = ep.scene_id[len("./data/scene_datasets/") :]
        absolute = Path(ep.scene_id)
        ep.scene_id = str(absolute.relative_to(absolute.parents[1]))

    scene_key = scene.stem
    #out_file = f"./data/datasets/pointnav/gibson/v1/all/content/{scene_key}.json.gz"
    out_file = EPISODES_DIR / f'{SPLIT}_ddppo' / 'content' / f'{scene_key}.json.gz'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes_dir', type=Path, required=True)   # input
    parser.add_argument('--split', required=True)
    parsed = parser.parse_args()

    SPLIT = parsed.split

    with open(GIBSON_QUALITY, 'r') as csv_f:
        quality = pd.read_csv(csv_f)

    with open(GIBSON_SPLITS, 'r') as csv_f:
        splits = pd.read_csv(csv_f)

    # get available scenes (train/test from Gibson fullplus)
    all_scenes = parsed.scenes_dir.glob('*.glb')
    scene_ids = set([scene.stem for scene in all_scenes])

    # find those that are 2+
    two_plus = quality.loc[quality['quality'] >= 2]
    two_plus_splits = splits.loc[splits['id'].isin(set(two_plus.scene_id))]

    # narrow those down to those from this split
    split_scenes = two_plus_splits.loc[two_plus_splits[parsed.split] == 1].id
    scenes = [parsed.scenes_dir / f'{str.strip(split_scene)}.glb' for split_scene in split_scenes]
    print(scenes)

    #scenes = glob.glob("./data/scene_datasets/gibson/*.glb")
    with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        for _ in pool.imap_unordered(_generate_fn, scenes):
            pbar.update()

    #with gzip.open(f"./data/datasets/pointnav/gibson/v1/all/all.json.gz", "wt") as f:
    with gzip.open(EPISODES_DIR / f'{SPLIT}_ddppo/{parsed.split}.json.gz', 'wt') as f:
        json.dump(dict(episodes=[]), f)
