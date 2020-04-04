Executable = run_gen_habitat_ppo_testscenes_2.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot && Eldar == True)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_gen_habitat_ppo_testscenes_2.log
Output=/u/nimit/logs/$(ClusterId)_gen_habitat_ppo_testscenes_2.out
Error=/u/nimit/logs/$(ClusterId)_gen_habitat_ppo_testscenes_2.err

Queue 1
