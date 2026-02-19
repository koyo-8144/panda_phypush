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

INTER_Q = np.array([
    1.443146213744816, 
    -0.4092480278977176, 
    -1.9449646612766016, 
    -2.4508971273203524, 
    0.6587622390985489, 
    2.9312011031092347, 
    0.19828951818909815
])

INTER2_Q = np.array([
    1.315057246822537, 
    -1.0260285342934183, 
    -2.139266043512445, 
    -2.416179822520206, 
    -0.46637658449014024, 
    3.2340388702551524, 
    -1.3069465030366156
])

PUSHSET_Q = np.array([
    1.75714665,
    -1.08043168, 
    -2.39465545, 
    -2.05762684,  
    0.50821188,  
    1.82329159,
    -1.7067362
])

TEST_Q = np.array([
    2.2647625646576617, 
    -0.7309278836584928, 
    -2.8560501611609204, 
    -2.1939660246999635, 
    0.8481742448442512, 
    1.6207939107285603, 
    -2.0477857253259777
])


INTER_POSE = np.array([
    [ 0.71129521, -0.44960746,  0.54027643,  0.48424884,],
    [-0.63427915, -0.74179191,  0.217753,   -0.2417092, ],
    [ 0.30286932, -0.49757275, -0.81282532,  0.16966554,],
    [ 0.,          0.,          0.,          1.        ]
 ])

PUSHSET_POSE = np.array([
    [ 0.00354444,  0.99922547, -0.03894451,  0.41308047,],
 [ 0.06143123,  0.03865324,  0.99736254, -0.01612842,],
 [ 0.99809538, -0.0059275,  -0.06124783,  0.07658673,],
 [ 0.,          0.,          0.,          1.,        ]
])

TEST_POSE = np.array([
    [ 0.00453696,  0.99974505, -0.02167942,  0.3875866, ],
    [ 0.05939407,  0.02137175,  0.99800577, -0.03924178,],
    [ 0.99821466, -0.00581554, -0.05928311,  0.07155698,],
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
        
        # 3. Move to the Saved Target
        # We use joint positions because it is the most deterministic way 
        # to reach a specific configuration without flipping the elbow.
        print("Moving to Saved Pushset Configuration...")
        # Grasp (width, speed, force, epsilon_inner, epsilon_outer)
        gripper.grasp(0, 0.2, 10, 0.04, 0.04)
        time.sleep(1.0)
        # q = panda_py.ik(INTER_POSE)
        # panda.move_to_joint_position(q)
        # panda.move_to_joint_position(INTER_Q)

        # panda.move_to_joint_position(INTER2_Q)

        # q = panda_py.ik(PUSHSET_POSE)
        # panda.move_to_joint_position(q)
        # panda.move_to_joint_position(PUSHSET_Q)

        # q = panda_py.ik(TEST_POSE)
        # panda.move_to_joint_position(q)

        panda.move_to_joint_position(TEST_Q)
        curr_q_list = panda.get_state().q
        curr_q_np = np.array(curr_q_list, dtype=np.float64)
        curr_q = curr_q_np.reshape((7, 1))
        print("curr_q shape: ", curr_q.shape)
        q = panda_py.ik(PUSHSET_POSE, q_init=curr_q)

        if q is None or np.isnan(q).any():
            print("❌ IK Failed! The pose is unreachable.")
            print("output q from ik: ", q)
            return

        q_final = q.reshape((7, 1))
        print(f"Moving to Pushset Joints:\n{q_final}")
        panda.move_to_joint_position(q_final, speed_factor=0.2)
        print("Successfully reached the target!")

        print("Successfully reached the target!")
        print(f"Current Joint Positions:\n{panda.q}")

        # print("Going back to Neutral (Start) Position...")
        # panda.move_to_start()

    except Exception as e:
        print(f"An error occurred: {e}")
        # If it fails due to limits, you might need to adjust PUSHSET_Q slightly.

if __name__ == "__main__":
    move_to_pushset()