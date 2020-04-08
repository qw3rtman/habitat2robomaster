Executable = run_dontcrash-depth-student_15.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot && Eldar == True)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_dontcrash-depth-student_15.log
Output=/u/nimit/logs/$(ClusterId)_dontcrash-depth-student_15.out
Error=/u/nimit/logs/$(ClusterId)_dontcrash-depth-student_15.err

Queue 1
