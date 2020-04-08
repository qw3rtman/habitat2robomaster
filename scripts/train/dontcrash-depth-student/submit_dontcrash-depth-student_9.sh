Executable = run_dontcrash-depth-student_9.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot && Eldar == True)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_dontcrash-depth-student_9.log
Output=/u/nimit/logs/$(ClusterId)_dontcrash-depth-student_9.out
Error=/u/nimit/logs/$(ClusterId)_dontcrash-depth-student_9.err

Queue 1
