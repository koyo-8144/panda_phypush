import logging
import time
import numpy as np
import panda_py
from panda_py import libfranka, constants
from matplotlib import pyplot as plt
from phypush_const import PUSHSET_POSE, PUSHSET_Q, INTER_Q, USE_IK, HOSTNAME

# 1. Configuration
username = 'cobotmakerspace'
password = 'cobotmakerspace'

# The specific joint configuration (radians)
# q: [J1, J2, J3, J4, J5, J6, J7]



def move_to_pushset():
    logging.basicConfig(level=logging.INFO)
    
    try:
        desk = panda_py.Desk(HOSTNAME, username, password)
        # desk.unlock()
        # desk.activate_fci()

        # 1. Connect to Robot
        print(f"Connecting to {HOSTNAME}...")
        panda = panda_py.Panda(HOSTNAME)
        gripper = libfranka.Gripper(HOSTNAME)
        
        # 2. Reset to Neutral first
        # This clears any previous weird states and gives us a clean starting point.
        print("Moving to Neutral (Start) Position...")
        #panda.move_to_start()
        
        # 3. Move to the Saved Target
        # We use joint positions because it is the most deterministic way 
        # to reach a specific configuration without flipping the elbow.
        print("Moving to Saved Pushset Configuration...")
        # gripper.move(width=0.03, speed=0.1)
        # Grasp (width, speed, force, epsilon_inner, epsilon_outer)
        gripper.grasp(0.05, 0.2, 10, 0.04, 0.04)
        time.sleep(1.0)

        if USE_IK:
            panda.move_to_joint_position(INTER_Q)
            # Get current state
            curr_q_list = panda.get_state().q
            curr_q_np = np.array(curr_q_list, dtype=np.float64)
            curr_q = curr_q_np.reshape((7, 1))
            # Get the current angle of Joint 7 (index 6)
            current_j7_angle = float(curr_q_np[6])
            print(f"curr_q shape: {curr_q.shape}")
            print(f"Solving IK with Joint 7 locked at {current_j7_angle:.3f} rad...")

            q = panda_py.ik(PUSHSET_POSE, q_init=curr_q, q_7=current_j7_angle)

            if q is None or np.isnan(q).any():
                print("❌ IK Failed with current J7. Trying ALL redundant solutions...")
                
                # --- FALLBACK: Use ik_full to get all 4 elbow configurations ---
                # ik_full returns a (4, 7) array. We can check if any of the 4 rows are valid.
                all_solutions = panda_py.ik_full(PUSHSET_POSE, q_init=curr_q, q_7=current_j7_angle)
                
                valid_q = None
                for i in range(4):
                    sol = all_solutions[i]
                    if not np.isnan(sol).any():
                        valid_q = sol
                        print(f"✅ Found valid configuration at index {i}!")
                        break
                
                if valid_q is None:
                    print("❌ ALL IK solutions failed. The Pose is physically unreachable (e.g., Z is too low, or it hits a joint limit).")
                    print("Target Z height:", PUSHSET_POSE[2, 3])
                    return
                else:
                    q = valid_q

            q_final = q.reshape((7, 1))
            print(f"Moving to Pushset Joints:\n{q_final.flatten()}")
            panda.move_to_joint_position(q_final, speed_factor=0.2)
            print("Successfully reached the target!")

        else:
            panda.move_to_joint_position(PUSHSET_Q, speed_factor=0.2)

        print("Successfully reached the target!")
        print(f"Current Joint Positions:\n{panda.q}")
        print(f"Current EE POSE:\n{panda.get_pose()}")

        # print("Going back to Neutral (Start) Position...")
        # panda.move_to_start()

    except Exception as e:
        print(f"An error occurred: {e}")
        # If it fails due to limits, you might need to adjust PUSHSET_Q slightly.

if __name__ == "__main__":
    move_to_pushset()