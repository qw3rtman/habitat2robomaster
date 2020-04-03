Executable = run_habitat_ppo_depth_2.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot && Eldar == True)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_habitat_ppo_depth_2.log
Output=/u/nimit/logs/$(ClusterId)_habitat_ppo_depth_2.out
Error=/u/nimit/logs/$(ClusterId)_habitat_ppo_depth_2.err

Queue 1
