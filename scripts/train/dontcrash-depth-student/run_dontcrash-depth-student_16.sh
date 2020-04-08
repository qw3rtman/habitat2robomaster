#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster/habitat2robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
python train_image_dagger.py \
    --resnet_model resnet18 \
    --dataset_dir /scratch/cluster/nimit/data/habitat/ppo_depth \
    --dataset_size 1.0 \
    --teacher_task dontcrash \
    --teacher_proxy depth \
    --dagger_dataset_dir /scratch/cluster/nimit/data/habitat/dontcrash-depth/resnet18-direct-50-1000-1.0-64-0.0001-5e-05 \
    --lr 0.0001 \
    --batch_size 64 \
    --weight_decay 5e-05 \
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \
    --dagger 
