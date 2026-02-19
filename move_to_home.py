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
    1.75714665,
    -1.08043168, 
    -2.39465545, 
    -2.05762684,  
    0.50821188,  
    1.82329159,
    -1.7067362
])

# The corresponding End-Effector Pose (Homogeneous Transformation Matrix)
# We store this for reference, but we will move using PUSHSET_Q for safety.
PUSHSET_POSE = np.array([
    [ 0.00354444,  0.99922547, -0.03894451,  0.41308047,],
 [ 0.06143123,  0.03865324,  0.99736254, -0.01612842,],
 [ 0.99809538, -0.0059275,  -0.06124783,  0.07658673,],
 [ 0.,          0.,          0.,          1.,        ]
])

def move_to_pushset():
    logging.basicConfig(level=logging.INFO)
    
    try:
        desk = panda_py.Desk(hostname, username, password)
        # desk.unlock()
        # desk.activate_fci()

        # 1. Connect to Robot
        print(f"Connecting to {hostname}...")
        panda = panda_py.Panda(hostname)
        gripper = libfranka.Gripper(hostname)
        
        # 2. Reset to Neutral first
        # This clears any previous weird states and gives us a clean starting point.
        print("Moving to Neutral (Start) Position...")
        panda.move_to_start()
        

    except Exception as e:
        print(f"An error occurred: {e}")
        # If it fails due to limits, you might need to adjust PUSHSET_Q slightly.

if __name__ == "__main__":
    move_to_pushset()