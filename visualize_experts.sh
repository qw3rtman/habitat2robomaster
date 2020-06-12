python -m buffer.target_dataset \
    --dataset_dir /scratch/cluster/nimit/data/habitat/replica-apartment_2-k=3/train/000005 \
    --source_teacher /scratch/cluster/nimit/models/apartment_0-semantic-teacher/model_latest.t7 \
    --goal_prediction /scratch/cluster/nimit/wandb/run-20200612_035526--2836192455627059422/model_030.t7
    #--goal_prediction /scratch/cluster/nimit/wandb/run-20200612_035525-3556452068915189436/model_020.t7
