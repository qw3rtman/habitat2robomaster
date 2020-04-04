#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster/habitat2robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
python habitat_wrapper.py \
    --input_type depth \
    --dataset_dir /scratch/cluster/nimit/data/habitat/ppo_depth  \
    --num_episodes 2000
