import logging
import time
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import panda_py
from panda_py import libfranka, controllers
from const import PUSHSET_POSE, PUSHSET_Q

# --- 1. Configuration ---
hostname = '172.22.2.3'
username = 'cobotmakerspace'
password = 'cobotmakerspace'

# Desired End-Effector Velocity in LOCAL Frame [vx, vy, vz, wx, wy, wz]
# [0.0, 0.08, 0.0] -> Now strictly moves in the GRIPPER'S Y-axis
V_DESIRED_LOCAL = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0]) 

# Duration parameters
VELOCITY_DURATION = 3.0 
DT = 0.01  # Fixed time step (1/100 Hz)


SMOOTHING_WINDOW = 5 

def ensure_dirs():
    cwd = os.getcwd()
    for folder in ['vis', 'csv_data']:
        path = os.path.join(cwd, folder)
        if not os.path.exists(path):
            os.makedirs(path)
    return cwd

def save_and_plot(history_vel, history_acc):
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
            row = [i, DT] + history_vel[i].tolist() + history_acc[i].tolist()
            writer.writerow(row)
    print(f"Data saved to {csv_path}")

    # --- Plotting ---
    labels = ['Local X', 'Local Y', 'Local Z', 'Local Wx', 'Local Wy', 'Local Wz']
    units = ['m/s']*3 + ['rad/s']*3
    acc_units = ['m/s^2']*3 + ['rad/s^2']*3
    
    np_vel = np.array(history_vel)
    np_acc = np.array(history_acc)
    steps = np.arange(len(history_vel))

    fig_vel, axes_vel = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
    fig_acc, axes_acc = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
    
    for i in range(6):
        axes_vel[i].plot(steps, np_vel[:, i], 'b-', label='Velocity')
        axes_vel[i].set_title(f"Velocity: {labels[i]}")
        axes_vel[i].set_ylabel(units[i])
        axes_vel[i].grid(True, alpha=0.3)
        
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
    
    history_vel = []
    history_acc = []
    vel_smoothing_buffer = []
    prev_vel_w = np.zeros(6)

    try:
        # --- Connect ---
        print(f"Connecting to {hostname}...")
        panda = panda_py.Panda(hostname)
        gripper = libfranka.Gripper(hostname)
        
        # --- Init ---
        print("Moving to Neutral...")
        panda.move_to_start()
        
        print("Closing Gripper...")
        gripper.grasp(0, 0.2, 10, 0.04, 0.04)
        time.sleep(1.0)
        
        print(f"Moving to Pushset Pose...")
        panda.move_to_joint_position(PUSHSET_Q, speed_factor=0.2)
        
        print("Waiting 2s for controller switch...")
        time.sleep(2.0)
        
        # --- Velocity Control ---
        print(f"Starting Local Velocity Control: {V_DESIRED_LOCAL}")
        
        ctrl = controllers.IntegratedVelocity()
        panda.start_controller(ctrl)
        model = panda.get_model()
        
        # Capture Initial Rotation (Start Frame)
        state0 = panda.get_state()
        O_T_EE_0 = np.array(state0.O_T_EE).reshape((4, 4), order='F')
        R_start = O_T_EE_0[:3, :3]
        R_start_T = R_start.T

        with panda.create_context(frequency=100.0) as ctx:
            start_time = time.time()
            
            while ctx.ok():
                if time.time() - start_time > VELOCITY_DURATION:
                    break
                
                state = panda.get_state()
                
                # --- 1. Jacobian Calc ---
                J_flat = model.zero_jacobian(libfranka.Frame.kEndEffector, state)
                J = np.array(J_flat).reshape((6, 7), order='F')
                J_pinv = np.linalg.pinv(J)
                
                # --- 2. NEW: Transform Command from Local -> Base Frame ---
                # Get current rotation to transform the Local Command Vector
                O_T_EE_curr = np.array(state.O_T_EE).reshape((4, 4), order='F')
                R_curr = O_T_EE_curr[:3, :3]
                
                # V_base = R_current * V_local
                cmd_lin_base = R_curr @ V_DESIRED_LOCAL[:3]
                cmd_ang_base = R_curr @ V_DESIRED_LOCAL[3:]
                V_CMD_BASE = np.concatenate([cmd_lin_base, cmd_ang_base])

                # Calculate Joint Velocities (dq = J_pinv * V_base)
                dq_cmd = J_pinv @ V_CMD_BASE
                ctrl.set_control(dq_cmd)
                
                # --- 3. Recording (Same as before) ---
                dq_actual = np.array(state.dq)
                vel_w_raw = J @ dq_actual
                
                vel_smoothing_buffer.append(vel_w_raw)
                if len(vel_smoothing_buffer) > SMOOTHING_WINDOW:
                    vel_smoothing_buffer.pop(0)
                vel_w_smoothed = np.mean(vel_smoothing_buffer, axis=0)
                
                # Accel (Finite Diff)
                acc_w = (vel_w_smoothed - prev_vel_w) / DT
                prev_vel_w = vel_w_smoothed.copy()
                
                # Transform RECORDED data back to Local Start Frame for Plotting
                lin_vel_local = R_start_T @ vel_w_smoothed[:3]
                ang_vel_local = R_start_T @ vel_w_smoothed[3:]
                vel_local = np.concatenate([lin_vel_local, ang_vel_local])
                
                lin_acc_local = R_start_T @ acc_w[:3]
                ang_acc_local = R_start_T @ acc_w[3:]
                acc_local = np.concatenate([lin_acc_local, ang_acc_local])
                
                history_vel.append(vel_local)
                history_acc.append(acc_local)

        print("Finished.")
        panda.stop_controller()
        
        print("Cleanup...")
        panda.move_to_start()
        gripper.move(0.08, 0.2)
        
        if len(history_vel) > 0:
            print("Saving...")
            save_and_plot(history_vel, history_acc)

    except Exception as e:
        print(f"Error: {e}")
        if len(history_vel) > 0:
            save_and_plot(history_vel, history_acc)

if __name__ == "__main__":
    run_push_and_velocity()