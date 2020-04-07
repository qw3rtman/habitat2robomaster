#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster/habitat2robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
python train_image_dagger.py \
    --network ddppo-direct \
    --input_type depth \
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \
    --dataset_dir /scratch/cluster/nimit/data/habitat/ppo_depth \
    --per_epoch 1000 \
    --seeded \
    --lr 0.0001 \
    --batch_size 64 \
    --weight_decay 5e-05
