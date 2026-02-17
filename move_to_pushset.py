import logging
import time
import numpy as np
import panda_py
from panda_py import libfranka, constants
from matplotlib import pyplot as plt

# 1. Configuration
hostname = '172.22.2.3'
username = 'cobotmakerspace'
password = 'cobotmakerspace'

# The specific joint configuration (radians)
# q: [J1, J2, J3, J4, J5, J6, J7]
PUSHSET_Q = np.array([
    1.72169, 
    -1.02605, 
    -2.27493, 
    -2.10522, 
    0.503725, 
    1.85432, 
    -1.62299
])

# The corresponding End-Effector Pose (Homogeneous Transformation Matrix)
# We store this for reference, but we will move using PUSHSET_Q for safety.
PUSHSET_POSE = np.array([
    [0.02510015,  0.99965103,  0.00696838,  0.44908469],
    [0.04269919, -0.00803622,  0.99905564, -0.05367641],
    [0.99876299, -0.0247789,  -0.04288682,  0.10449314],
    [0.,          0.,          0.,          1.        ]
])

def move_to_pushset():
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 1. Connect to Robot
        print(f"Connecting to {hostname}...")
        panda = panda_py.Panda(hostname)
        gripper = libfranka.Gripper(hostname)
        
        # 2. Reset to Neutral first
        # This clears any previous weird states and gives us a clean starting point.
        print("Moving to Neutral (Start) Position...")
        panda.move_to_start()
        
        # 3. Move to the Saved Target
        # We use joint positions because it is the most deterministic way 
        # to reach a specific configuration without flipping the elbow.
        print("Moving to Saved Pushset Configuration...")
        # Grasp (width, speed, force, epsilon_inner, epsilon_outer)
        gripper.grasp(0, 0.2, 10, 0.04, 0.04)
        time.sleep(1)
        # q = panda_py.ik(PUSHSET_POSE)
        # panda.move_to_joint_position(q, speed_factor=0.2)
        panda.move_to_joint_position(PUSHSET_Q)
      
        print("Successfully reached the target!")
        print(f"Current Joint Positions:\n{panda.q}")

        # print("Going back to Neutral (Start) Position...")
        # panda.move_to_start()

    except Exception as e:
        print(f"An error occurred: {e}")
        # If it fails due to limits, you might need to adjust PUSHSET_Q slightly.

if __name__ == "__main__":
    move_to_pushset()