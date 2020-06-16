python -m buffer.target_dataset \
    --dataset_dir /scratch/cluster/nimit/data/habitat/replica-frl_apartment_4-k=3/train/000011 \
    --scene frl_apartment_4 \
    --source_teacher /scratch/cluster/nimit/models/apartment_0-semantic-teacher/002/model_030.t7 \
    --goal_prediction /scratch/cluster/nimit/checkpoints/resnet34_0.0001_5e-05_64_10.0_semantic_6.14-v1/model_latest.t7
    #--dataset_dir /scratch/cluster/nimit/data/habitat/replica-apartment_2/train/000081 \
    #--dataset_dir /scratch/cluster/nimit/data/habitat/replica-apartment_0-k=3-target/train/000001 \
    #--source_teacher /scratch/cluster/nimit/models/apartment_0-semantic-teacher/model_latest.t7 \
    #--goal_prediction /scratch/cluster/nimit/checkpoints/resnet34_0.0001_5e-05_128_1.0_semantic_6.14-v1/model_latest.t7
    #--goal_prediction /scratch/cluster/nimit/wandb/run-20200614_023413-4872519130998882165/model_010.t7
    #--goal_prediction /scratch/cluster/nimit/wandb/run-20200612_035525-3556452068915189436/model_020.t7
    #--goal_prediction /scratch/cluster/nimit/wandb/run-20200612_035526--2836192455627059422/model_100.t7
    #--goal_prediction /scratch/cluster/nimit/wandb/run-20200612_035526--2836192455627059422/model_030.t7
