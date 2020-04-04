#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster/habitat2robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
python habitat_wrapper.py \
    --input_type rgb \
    --dataset_dir /scratch/cluster/nimit/data/habitat/ppo_rgb  \
    --num_episodes 1500
