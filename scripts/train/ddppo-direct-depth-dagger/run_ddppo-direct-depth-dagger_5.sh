#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster/habitat2robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
python train_image_dagger.py \
        --network ddppo-direct \
        --input_type depth \
        --per_epoch 5000 \
        --checkpoint_dir /scratch/cluster/nimit/checkpoints \
        --dataset_dir /scratch/cluster/nimit/data/habitat2/ddppo-direct-depth-dagger-4 \
        --lr 0.0001 \
        --batch_size 128 \
        --weight_decay 5e-05
    