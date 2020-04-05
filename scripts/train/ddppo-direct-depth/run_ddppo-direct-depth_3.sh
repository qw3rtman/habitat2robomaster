#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster/habitat2robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
python train_image.py \
    --network ddppo-direct \
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \
    --dataset_dir /scratch/cluster/nimit/data/habitat/ppo_depth \
    --resnet_model resnet34 \
    --lr 0.001 \
    --batch_size 64 \
    --weight_decay 5e-05
