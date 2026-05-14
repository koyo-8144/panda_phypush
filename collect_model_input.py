import logging
import time
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import panda_py
from panda_py import libfranka, controllers
from pathlib import Path
from phypush_const import PUSHSET_POSE, PUSHSET_Q, HOSTNAME

# ==========================================
# 1. CONFIGURATION
# ==========================================
username = 'cobotmakerspace'
password = 'cobotmakerspace'

V_DESIRED_BASE_SET = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]) 
V_DESIRED_BASE_PUSH = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0]) # Move 8 cm/s 
PUSH_AXIS_IDX = 1  # 0=X, 1=Y, 2=Z 

# Duration parameters
SET_VELOCITY_DURATION = 3.0 
PUSH_VELOCITY_DURATION = 7.0 
DT = 0.01  # Fixed time step (1/100 Hz)

# --- OFFLINE DATA COLLECTION PARAMS ---
SMOOTHING_WINDOW = 1  

# ==========================================
# DATA PROCESSING & SAVING
# ==========================================
def process_and_save_data(history_vel, history_acc, run_id, m_gt, mu_gt, obj_name, surface_name, save_dir="."):
    """
    1. Finds the impact peak.
    2. Extracts the 60/100 step windows.
    3. Saves FULL sequence CSV.
    4. Saves 100-step context CSV.
    5. Saves single Ground Truth CSV (if it doesn't exist).
    6. Saves Metadata CSV and verification plots.
    """

    # --- Configuration ---
    T_BEFORE = -3
    T_AFTER = 57 
    WINDOW_LEN = T_AFTER - T_BEFORE  
    CTX_LEN = 100  
    PAD_BEFORE_INF = 20  

    np_vel_full = np.array(history_vel)
    np_acc_full = np.array(history_acc)
    total_steps = len(np_vel_full)

    if total_steps < CTX_LEN:
        print(f"⚠️ Not enough data (Total Steps: {total_steps}). Need at least {CTX_LEN}.")
        return

    # 1. Find Impact Peak
    acc_push_axis = np_acc_full[:, PUSH_AXIS_IDX]
    vel_push_axis = np_vel_full[:, PUSH_AXIS_IDX]
    
    skip_start = 100
    skip_end = 100
    
    if total_steps > (skip_start + skip_end):
        search_window_vel = vel_push_axis[skip_start : -skip_end]
        t_peak_vel = int(np.argmin(search_window_vel)) + skip_start

        acc_search_radius = 20
        local_start = max(0, t_peak_vel - acc_search_radius)
        local_end = min(total_steps, t_peak_vel + acc_search_radius)

        local_search_window_acc = acc_push_axis[local_start:local_end]
        t_peak_acc = int(np.argmin(local_search_window_acc)) + local_start

        t_peak = t_peak_acc
        print(f"t_peak_vel (Velocity Drop): {t_peak_vel}")
        print(f"t_peak_acc (Impact Spike):  {t_peak_acc}")
    else:
        t_peak = int(np.argmin(acc_push_axis))
        
    print(f"--- Final Anchored Impact Peak at Step {t_peak} ---")

    # 2. Define Inference Indices
    inf_start_abs = max(0, t_peak + T_BEFORE)
    inf_end_abs = inf_start_abs + WINDOW_LEN
    
    if inf_end_abs > total_steps:
        shift = inf_end_abs - total_steps
        inf_start_abs -= shift
        inf_end_abs -= shift

    # 3. Extract Context Window
    ctx_start_abs = max(0, inf_start_abs - PAD_BEFORE_INF)
    ctx_end_abs = ctx_start_abs + CTX_LEN
    
    if ctx_end_abs > total_steps:
        ctx_end_abs = total_steps
        ctx_start_abs = max(0, ctx_end_abs - CTX_LEN)

    vel_100 = np_vel_full[ctx_start_abs:ctx_end_abs]
    acc_100 = np_acc_full[ctx_start_abs:ctx_end_abs]

    csv_inf_start = inf_start_abs - ctx_start_abs
    csv_inf_end = inf_end_abs - ctx_start_abs

    # --- PREPARE DIRECTORIES ---
    condition_folder = f"{obj_name}_{surface_name}"
    
    vis_dir = os.path.join(save_dir, "vis", "offline_collection", condition_folder)
    csv_dir = os.path.join(save_dir, "csv_data", "offline_collection", condition_folder)
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S") 
    base_filename = f"iter{run_id}_{timestamp}"

    # ==========================================
    # --- SAVE CSV 1: FULL SEQUENCE ---
    # ==========================================
    full_csv_path = os.path.join(csv_dir, f"{base_filename}_full.csv")
    with open(full_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'v_y', 'a_y', 'is_context_region', 'is_inference_region'])
        for i in range(total_steps):
            is_ctx = 1 if (i >= ctx_start_abs and i < ctx_end_abs) else 0
            is_inf = 1 if (i >= inf_start_abs and i < inf_end_abs) else 0
            writer.writerow([i, np_vel_full[i, PUSH_AXIS_IDX], np_acc_full[i, PUSH_AXIS_IDX], is_ctx, is_inf])

    # ==========================================
    # --- SAVE CSV 2: 100-STEP CONTEXT ---
    # ==========================================
    ctx_csv_path = os.path.join(csv_dir, f"{base_filename}_100steps.csv")
    with open(ctx_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step_local', 'v_y', 'a_y', 'is_inference_region'])
        for i in range(len(vel_100)):
            is_inf = 1 if (i >= csv_inf_start and i < csv_inf_end) else 0
            writer.writerow([i, vel_100[i, PUSH_AXIS_IDX], acc_100[i, PUSH_AXIS_IDX], is_inf])
    
    # ==========================================
    # --- SAVE CSV 3: GROUND TRUTH (ONCE) ---
    # ==========================================
    gt_csv_path = os.path.join(csv_dir, "ground_truth.csv")
    if not os.path.exists(gt_csv_path):
        with open(gt_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Value"])  
            writer.writerow(["M_GT", m_gt])
            writer.writerow(["MU_GT", mu_gt])
        print(f"📝 Created unified Ground Truth file: {gt_csv_path}")

    # ==========================================
    # --- SAVE METADATA CSV ---
    # ==========================================
    metadata = {
        "RUN_ID": run_id,
        "TIMESTAMP": timestamp,
        "OBJECT": obj_name,
        "SURFACE": surface_name,
        "SMOOTHING_WINDOW": SMOOTHING_WINDOW,
        "TOTAL_STEPS": total_steps,
        "GLOBAL_IMPACT_STEP": t_peak,
        "CTX_START_ABS": ctx_start_abs,
        "CTX_END_ABS": ctx_end_abs,
        "INF_START_ABS": inf_start_abs,
        "INF_END_ABS": inf_end_abs
    }

    metadata_csv_path = os.path.join(csv_dir, f"{base_filename}_metadata.csv")
    with open(metadata_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])  
        for key, value in metadata.items():
            writer.writerow([key, value])

    # ==========================================
    # --- PLOT VERIFICATION ---
    # ==========================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Offline Data Collection [Run {run_id}] | Object: {obj_name} | Surface: {surface_name}", fontsize=14, fontweight='bold')

    steps_full = np.arange(total_steps)
    steps_60 = np.arange(inf_start_abs, inf_end_abs)

    axes[0].plot(steps_full, np_vel_full[:, PUSH_AXIS_IDX], 'b-', alpha=0.3, label=f'Full Velocity')
    axes[0].plot(steps_60, np_vel_full[inf_start_abs:inf_end_abs, PUSH_AXIS_IDX], 'b-', linewidth=3, label=f'60-Step Window')
    axes[0].axvline(ctx_start_abs, color='g', linestyle='--', alpha=0.5, label='Context Window Start')
    axes[0].set_title("Velocity Output")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps_full, np_acc_full[:, PUSH_AXIS_IDX], 'r-', alpha=0.3, label=f'Full Acceleration')
    axes[1].plot(steps_60, np_acc_full[inf_start_abs:inf_end_abs, PUSH_AXIS_IDX], 'r-', linewidth=3, label=f'60-Step Window')
    axes[1].axvline(t_peak, color='k', linestyle='--', alpha=0.8, label='Impact Peak (Min)')
    axes[1].set_title(f"Acceleration: Peak at Step {t_peak}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path_full = os.path.join(vis_dir, f"{base_filename}_verification.png")
    plt.savefig(plot_path_full)
    plt.close()

    print(f"\n✅ COLLECTION SUCCESSFUL (Run {run_id})!")
    print(f"📂 Saved to: {csv_dir}")

# ==========================================
# MAIN LOOP
# ==========================================
def run_push_and_collect(run_id="manual", m_gt="None", mu_gt="None", obj_name="unknown", surface_name="unknown"):
    logging.basicConfig(level=logging.INFO)
    
    history_vel = []
    history_acc = []
    vel_smoothing_buffer = []
    prev_vel_w = np.zeros(6)

    try:
        print(f"Connecting to {HOSTNAME}...")
        panda = panda_py.Panda(HOSTNAME)
        gripper = libfranka.Gripper(HOSTNAME)
        
        print(f"Moving to Pushset Pose...")
        panda.move_to_joint_position(PUSHSET_Q, speed_factor=0.2)
        if "colored_cubes" in obj_name:
            gripper.move(width=0.06, speed=0.1)
        elif "blue_cylinder" in obj_name:
            gripper.move(width=0.06, speed=0.1)
        elif "wooden_cube_even" in obj_name:
            gripper.move(width=0.06, speed=0.1)
        else:
            state = gripper.read_once()
            gripper.move(width=state.max_width, speed=0.1)
        time.sleep(2.0)
        
        # START the Velocity Controller for the set motion
        ctrl = controllers.IntegratedVelocity()
        panda.start_controller(ctrl)
        model_robot = panda.get_model()

        with panda.create_context(frequency=1/DT) as ctx_set:
            start_time_set = time.time()
            
            while ctx_set.ok():
                if time.time() - start_time_set > SET_VELOCITY_DURATION:
                    break
                
                state = panda.get_state()
                J_flat = model_robot.zero_jacobian(libfranka.Frame.kEndEffector, state)
                J = np.array(J_flat).reshape((6, 7), order='F')
                J_pinv = np.linalg.pinv(J)
                
                dq_cmd = J_pinv @ V_DESIRED_BASE_SET
                ctrl.set_control(dq_cmd)
                
        # STOP the controller before joint motion!
        panda.stop_controller()
        time.sleep(3.0)

        # Move slightly to final pushset
        print(f"Resetting Pushset Pose...")
        panda.move_to_joint_position(PUSHSET_Q, speed_factor=0.2)
        time.sleep(0.5)
        
        # RESTART the velocity controller for the push!
        panda.start_controller(ctrl)
        
        print(f"Executing Push (Run {run_id})...")
        with panda.create_context(frequency=1/DT) as ctx_push:
            start_time_push = time.time()
            
            while ctx_push.ok():
                if time.time() - start_time_push > PUSH_VELOCITY_DURATION:
                    break
                
                state = panda.get_state()
                J_flat = model_robot.zero_jacobian(libfranka.Frame.kEndEffector, state)
                J = np.array(J_flat).reshape((6, 7), order='F')
                J_pinv = np.linalg.pinv(J)
                
                dq_cmd = J_pinv @ V_DESIRED_BASE_PUSH
                ctrl.set_control(dq_cmd)
                
                dq_actual = np.array(state.dq)
                vel_w_raw = J @ dq_actual
                
                # Smooth
                vel_smoothing_buffer.append(vel_w_raw)
                if len(vel_smoothing_buffer) > SMOOTHING_WINDOW:
                    vel_smoothing_buffer.pop(0)
                vel_w_smoothed = np.mean(vel_smoothing_buffer, axis=0)
                
                # Accel
                acc_w = (vel_w_smoothed - prev_vel_w) / DT
                prev_vel_w = vel_w_smoothed.copy()
                
                history_vel.append(vel_w_smoothed)
                history_acc.append(acc_w)
        
        panda.move_to_start()
                
        # Final cleanup
        panda.stop_controller()
        print("Motion finished. Processing data...")
        
        process_and_save_data(history_vel, history_acc, run_id=run_id, m_gt=m_gt, mu_gt=mu_gt, obj_name=obj_name, surface_name=surface_name)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect robot push data.")
    parser.add_argument("--run_id", type=str, default="manual", help="Iteration ID")
    parser.add_argument("--m_gt", type=str, default="None", help="Ground truth mass")
    parser.add_argument("--mu_gt", type=str, default="None", help="Ground truth friction")
    parser.add_argument("--object", type=str, default="unknown", help="Object name")
    parser.add_argument("--surface", type=str, default="unknown", help="Surface name")
    args = parser.parse_args()
    
    run_push_and_collect(run_id=args.run_id, m_gt=args.m_gt, mu_gt=args.mu_gt, obj_name=args.object, surface_name=args.surface)