import logging
import time
import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import panda_py
from panda_py import libfranka, controllers
from pathlib import Path
from phypush_transformer import PhysicsTransformerEstimator
from const import PUSHSET_POSE, PUSHSET_Q, HOSTNAME

# ==========================================
# 1. CONFIGURATION
# ==========================================
username = 'cobotmakerspace'
password = 'cobotmakerspace'


V_DESIRED_BASE_PUSH = np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0])

PUSH_VELOCITY_DURATION = 2.0
DT = 0.01  # Fixed time step (1/100 Hz)



def run_push_and_velocity():
    logging.basicConfig(level=logging.INFO)
    
    
    try:
        print(f"Connecting to {HOSTNAME}...")
        panda = panda_py.Panda(HOSTNAME)
        gripper = libfranka.Gripper(HOSTNAME)
        
        print(f"Moving to Pushset Pose...")
        panda.move_to_joint_position(PUSHSET_Q, speed_factor=0.2)
        time.sleep(2.0)
        
        # START the Velocity Controller for the first motion
        ctrl = controllers.IntegratedVelocity()
        panda.start_controller(ctrl)
        model_robot = panda.get_model()
        
        with panda.create_context(frequency=1/DT) as ctx_push:
            start_time_push = time.time()
            
            while ctx_push.ok():
                if time.time() - start_time_push > PUSH_VELOCITY_DURATION:
                    break
                
                state = panda.get_state()
                J_flat = model_robot.zero_jacobian(libfranka.Frame.kEndEffector, state)
                J = np.array(J_flat).reshape((6, 7), order='F')
                J_pinv = np.linalg.pinv(J)
                
                # Command
                dq_cmd = J_pinv @ V_DESIRED_BASE_PUSH
                ctrl.set_control(dq_cmd)
                
                
        # Final cleanup
        panda.stop_controller()
        

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_push_and_velocity()