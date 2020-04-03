import habitat
import numpy as np
import cv2

# Load embodied AI task (PointNav) and a pre-specified virtual robot
env = habitat.Env(
    config=habitat.get_config("configs/datasets/pointnav/gibson.yaml")
)

observations = env.reset()

# Step through environment with random actions
while not env.episode_over:
    observations = env.step(env.action_space.sample())
    cv2.imshow('rgb', np.uint8(observations['rgb']))
    cv2.imshow('seg', np.uint8(observations['semantic']))
    cv2.waitKey(0)
