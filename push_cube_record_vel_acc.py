import logging
import time
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import panda_py
from panda_py import libfranka, controllers

# --- 1. Configuration ---
hostname = '172.22.2.3'
username = 'cobotmakerspace'
password = 'cobotmakerspace'

# Desired End-Effector Velocity [vx, vy, vz, wx, wy, wz]
# Example: Move +Y
V_DESIRED_BASE = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0]) 

# Duration to apply the velocity (seconds)
VELOCITY_DURATION = 3.0 
DT = 0.01  # Fixed time step (1/100 Hz)

# The specific joint configuration (radians)
PUSHSET_Q = np.array([
    1.72169, -1.02605, -2.27493, -2.10522, 0.503725, 1.85432, -1.62299
])

# Recording Parameters
SMOOTHING_WINDOW = 10  # Moving average window size

def ensure_dirs():
    cwd = os.getcwd()
    for folder in ['vis', 'csv_data']:
        path = os.path.join(cwd, folder)
        if not os.path.exists(path):
            os.makedirs(path)
    return cwd

def save_and_plot(history_vel, history_acc):
    """
    Saves data to CSV and plots 6-axis history.
    """
    cwd = ensure_dirs()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # --- Save CSV ---
    csv_path = os.path.join(cwd, "csv_data", f"push_data_{timestamp}.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['step', 'dt', 
                  'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z',
                  'a_x', 'a_y', 'a_z', 'aw_x', 'aw_y', 'aw_z']
        writer.writerow(header)
        
        for i in range(len(history_vel)):
            # Flatten arrays for CSV writing
            row = [i, DT] + history_vel[i].tolist() + history_acc[i].tolist()
            writer.writerow(row)
    
    print(f"Data saved to {csv_path}")

    # --- Plotting ---
    labels = ['Local X', 'Local Y', 'Local Z', 'Local Wx', 'Local Wy', 'Local Wz']
    units = ['m/s']*3 + ['rad/s']*3
    acc_units = ['m/s^2']*3 + ['rad/s^2']*3
    
    # Convert lists to numpy for plotting
    np_vel = np.array(history_vel)
    np_acc = np.array(history_acc)
    steps = np.arange(len(history_vel))

    fig_vel, axes_vel = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
    fig_acc, axes_acc = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
    
    for i in range(6):
        # Velocity
        axes_vel[i].plot(steps, np_vel[:, i], 'b-', label='Velocity')
        axes_vel[i].set_title(f"Velocity: {labels[i]}")
        axes_vel[i].set_ylabel(units[i])
        axes_vel[i].grid(True, alpha=0.3)
        
        # Acceleration
        axes_acc[i].plot(steps, np_acc[:, i], 'r-', label='Acceleration')
        axes_acc[i].set_title(f"Acceleration: {labels[i]}")
        axes_acc[i].set_ylabel(acc_units[i])
        axes_acc[i].grid(True, alpha=0.3)

    axes_vel[5].set_xlabel("Step")
    axes_acc[5].set_xlabel("Step")
    
    fig_vel.tight_layout()
    fig_acc.tight_layout()
    
    fig_vel.savefig(os.path.join(cwd, "vis", f"velocity_{timestamp}.png"))
    fig_acc.savefig(os.path.join(cwd, "vis", f"acceleration_{timestamp}.png"))
    print("Plots saved to /vis folder.")


def run_push_and_velocity():
    logging.basicConfig(level=logging.INFO)
    
    # Buffers for recording
    history_vel = []
    history_acc = []
    vel_smoothing_buffer = []
    
    # Previous state for acceleration calc
    prev_vel_w = np.zeros(6)

    try:
        # --- Connect ---
        print(f"Connecting to {hostname}...")
        panda = panda_py.Panda(hostname)
        gripper = libfranka.Gripper(hostname)
        
        # --- Step 1: Initialize ---
        print("Moving to Neutral (Start) Position...")
        panda.move_to_start()
        
        # --- Step 2: Grasp and Move to Pushset ---
        print("Closing Gripper...")
        gripper.grasp(0, 0.2, 10, 0.04, 0.04)
        time.sleep(1.0)
        
        print(f"Moving to Pushset Pose: {PUSHSET_Q}")
        panda.move_to_joint_position(PUSHSET_Q, speed_factor=0.2)
        print("Reached Pushset Pose.")
        
        # --- CRITICAL SAFETY PAUSE ---
        print("Waiting 2 seconds for controller switch...")
        time.sleep(2.0)
        
        # --- Step 3: Velocity Control & Recording ---
        print(f"Starting Velocity Control: {V_DESIRED_BASE} for {VELOCITY_DURATION}s")
        
        ctrl = controllers.IntegratedVelocity()
        panda.start_controller(ctrl)
        model = panda.get_model()
        
        # Capture Orientation Matrix (R_local) at the start of the push
        # We use this to transform World Velocities -> Local Push Frame
        state0 = panda.get_state()
        O_T_EE_0 = np.array(state0.O_T_EE).reshape((4, 4), order='F')
        R_local = O_T_EE_0[:3, :3]  # 3x3 Rotation Matrix
        R_local_T = R_local.T       # Transpose = Inverse for Rotation Matrices

        # Run control loop at exactly 100Hz
        with panda.create_context(frequency=100.0) as ctx:
            start_time = time.time()
            
            while ctx.ok():
                # Check timeout
                if time.time() - start_time > VELOCITY_DURATION:
                    break
                
                # --- A. Control Logic ---
                state = panda.get_state()
                J_flat = model.zero_jacobian(libfranka.Frame.kEndEffector, state)
                J = np.array(J_flat).reshape((6, 7), order='F')
                J_pinv = np.linalg.pinv(J)
                
                # Send Command
                dq_cmd = J_pinv @ V_DESIRED_BASE
                ctrl.set_control(dq_cmd)
                
                # --- B. Recording Logic (Fixed DT = 0.01) ---
                
                # 1. Get Real Joint Velocities (dq_actual)
                dq_actual = np.array(state.dq)
                
                # 2. Compute Cartesian Velocity in World Frame (V_w = J * dq)
                vel_w_raw = J @ dq_actual # [vx, vy, vz, wx, wy, wz]
                
                # 3. Smoothing (Moving Average)
                vel_smoothing_buffer.append(vel_w_raw)
                if len(vel_smoothing_buffer) > SMOOTHING_WINDOW:
                    vel_smoothing_buffer.pop(0)
                vel_w_smoothed = np.mean(vel_smoothing_buffer, axis=0)
                
                # 4. Compute Acceleration in World Frame (Finite Difference with Fixed DT)
                acc_w = (vel_w_smoothed - prev_vel_w) / DT
                prev_vel_w = vel_w_smoothed.copy()
                
                # # 5. Transform to Local Start Frame (Rotation only)
                # # Apply rotation to Linear ([:3]) and Angular ([3:]) parts separately
                # lin_vel_local = R_local_T @ vel_w_smoothed[:3]
                # ang_vel_local = R_local_T @ vel_w_smoothed[3:]
                # vel_local = np.concatenate([lin_vel_local, ang_vel_local])
                
                # lin_acc_local = R_local_T @ acc_w[:3]
                # ang_acc_local = R_local_T @ acc_w[3:]
                # acc_local = np.concatenate([lin_acc_local, ang_acc_local])

                vel_local = vel_w_smoothed
                acc_local = acc_w 
                
                # 6. Store
                history_vel.append(vel_local)
                history_acc.append(acc_local)

        print("Velocity motion finished.")
        panda.stop_controller()

        # --- Step 4: Cleanup ---
        print("Returning to Neutral (Start) Position...")
        panda.move_to_start()
        gripper.move(0.08, 0.2)
        
        # --- Step 5: Save and Visualize ---
        if len(history_vel) > 0:
            print("Processing data...")
            save_and_plot(history_vel, history_acc)
        else:
            print("No data recorded.")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Attempt to save whatever data we captured before crash
        if len(history_vel) > 0:
            print("Emergency Save of captured data...")
            save_and_plot(history_vel, history_acc)

if __name__ == "__main__":
    run_push_and_velocity()