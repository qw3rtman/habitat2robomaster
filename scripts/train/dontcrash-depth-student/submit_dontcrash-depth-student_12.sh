Executable = run_dontcrash-depth-student_12.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot && Eldar == True)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_dontcrash-depth-student_12.log
Output=/u/nimit/logs/$(ClusterId)_dontcrash-depth-student_12.out
Error=/u/nimit/logs/$(ClusterId)_dontcrash-depth-student_12.err

Queue 1
