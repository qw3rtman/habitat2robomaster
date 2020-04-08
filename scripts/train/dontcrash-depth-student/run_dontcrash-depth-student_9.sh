#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster/habitat2robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
python train_image_dagger.py \
    --resnet_model se_resneXt50 \
    --dataset_dir /scratch/cluster/nimit/data/habitat/ppo_depth \
    --dataset_size 1.0 \
    --teacher_task dontcrash \
    --teacher_proxy depth \
    --dagger_dataset_dir /scratch/cluster/nimit/data/habitat/dontcrash-depth/se_resneXt50-direct-50-1000-1.0-128-0.0001-5e-05 \
    --lr 0.0001 \
    --batch_size 128 \
    --weight_decay 5e-05 \
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \
    --dagger 
