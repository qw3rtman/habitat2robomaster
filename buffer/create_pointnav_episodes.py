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

NUM_EPISODES, EPISODES_DIR, SPLIT = 0, '', ''

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
        absolute = Path(ep.scene_id)
        ep.scene_id = str(absolute.relative_to(absolute.parents[3])) # 2 if mesh.ply

    scene_key = scene.parts[-3] # NOTE: replica specific, -2 if mesh.ply
    print(scene_key)
    out_file = EPISODES_DIR / f'{SPLIT}' / 'content' / f'{scene_key}.json.gz'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes_dir', type=Path, required=True)
    parser.add_argument('--episodes_dir', type=Path, required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--num_episodes', type=int, required=True)
    parsed = parser.parse_args()

    EPISODES_DIR = parsed.episodes_dir
    SPLIT = parsed.split
    NUM_EPISODES = parsed.num_episodes

    # get available scenes
    #scenes = set([scene/'mesh.ply' for scene in parsed.scenes_dir.iterdir()])
    scenes = set([scene/'habitat/mesh_semantic.ply' for scene in parsed.scenes_dir.iterdir()])
    print(scenes)

    with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        for _ in pool.imap_unordered(_generate_fn, scenes):
            pbar.update()

    with gzip.open(parsed.episodes_dir / f'{parsed.split}/{parsed.split}.json.gz', 'wt') as f:
        json.dump(dict(episodes=[]), f)
