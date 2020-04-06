Executable = run_ddppo-direct-depth-dagger_2.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot && Eldar == True)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_ddppo-direct-depth-dagger_2.log
Output=/u/nimit/logs/$(ClusterId)_ddppo-direct-depth-dagger_2.out
Error=/u/nimit/logs/$(ClusterId)_ddppo-direct-depth-dagger_2.err

Queue 1
