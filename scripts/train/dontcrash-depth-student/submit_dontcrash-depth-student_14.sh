Executable = run_dontcrash-depth-student_14.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot && Eldar == True)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_dontcrash-depth-student_14.log
Output=/u/nimit/logs/$(ClusterId)_dontcrash-depth-student_14.out
Error=/u/nimit/logs/$(ClusterId)_dontcrash-depth-student_14.err

Queue 1
