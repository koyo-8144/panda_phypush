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

# Activate information log level as panda-py is chatty
logging.basicConfig(level=logging.INFO)

def run_panda_test():
    try:
        # 2. Desk Connection (Code Block 1)
        # Unlock brakes and activate FCI for control
        print("\n--- Connecting to Desk ---")
        desk = panda_py.Desk(hostname, username, password)
        # desk.unlock()
        # desk.activate_fci()

        # 3. Robot and Gripper Initialization (Code Block 2)
        print("\n--- Initializing Robot and Gripper ---")
        panda = panda_py.Panda(hostname)
        gripper = libfranka.Gripper(hostname)

        # 4. Joint Space Motion (Code Block 3)
        print("\n--- Moving in Joint Space ---")
        panda.move_to_start()
        pose = panda.get_pose()
        # Lower z-coordinate by 0.1m
        pose[2, 3] -= 0.1
        # Calculate Inverse Kinematics and move
        q = panda_py.ik(pose)
        panda.move_to_joint_position(q)

        # 5. Cartesian Space Motion (Code Block 4)
        print("\n--- Moving in Cartesian Space ---")
        panda.move_to_start()
        pose = panda.get_pose()
        pose[2, 3] -= 0.1
        # Move directly to the pose matrix
        panda.move_to_pose(pose)

        # 6. Gripper and State Info (Inline Examples)
        print("\n--- Robot State and Gripper Test ---")
        print(f"Current State: {panda.get_state()}")
        print(f"Model Info: {panda.get_model()}")
        
        # Grasp (width, speed, force, epsilon_inner, epsilon_outer)
        gripper.grasp(0, 0.2, 10, 0.04, 0.04)
        time.sleep(1)
        # Open gripper to 8cm
        gripper.move(0.08, 0.2)

        # 7. Logging and Trajectory Comparison (Code Block 5)
        print("\n--- Running Logged Trajectories ---")
        T_0 = panda_py.fk(constants.JOINT_POSITION_START)
        T_0[1, 3] = 0.25  # Move to the left
        T_1 = T_0.copy()
        T_1[1, 3] = -0.25 # Move to the right

        # Move to starting pose
        panda.move_to_pose(T_0)

        # Log Cartesian Motion
        print("Logging Cartesian Motion...")
        panda.enable_logging(40000)
        panda.move_to_pose(T_1, speed_factor=0.01,
                           stiffness=2 * np.array([600, 600, 600, 600, 250, 150, 50]))
        panda.disable_logging()
        cartesian_log = panda.get_log()

        # Reset and Log Joint Motion
        panda.move_to_pose(T_0)
        print("Logging Joint Motion...")
        panda.enable_logging(10000)
        panda.move_to_joint_position(panda_py.ik(T_1))
        panda.disable_logging()
        joint_log = panda.get_log()

        # 8. Visualization
        print("\n--- Generating Plots ---")
        def plot_path(log_data, ax, title):
            # Column 13 is Y, Column 14 is Z in the flat O_T_EE matrix
            data = np.array(log_data['O_T_EE'])
            ax.plot(data[:, 13], data[:, 14])
            ax.set_xlim(-0.3, 0.3)
            ax.set_ylim(0.25, 0.75)
            ax.set_xlabel('y (m)')
            ax.set_ylabel('z (m)')
            ax.grid(True)
            ax.set_title(title)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
        plot_path(cartesian_log, ax0, 'Cartesian Motion Path')
        plot_path(joint_log, ax1, 'Joint Motion Path')
        fig.tight_layout()
        plt.show()

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run_panda_test()