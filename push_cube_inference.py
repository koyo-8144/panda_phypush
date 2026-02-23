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
from const import PUSHSET_POSE, PUSHSET_Q

# ==========================================
# 1. CONFIGURATION
# ==========================================
# hostname = '172.22.2.3'
hostname = '172.22.2.4'
username = 'cobotmakerspace'
password = 'cobotmakerspace'

# COMMAND: Move 8 cm/s along Base Y-Axis (Index 1)
V_DESIRED_BASE = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0]) 
PUSH_AXIS_IDX = 1  # 0=X, 1=Y, 2=Z (Must match non-zero element in V_DESIRED)

# Duration parameters
VELOCITY_DURATION = 7.0 
DT = 0.01  # Fixed time step (1/100 Hz)

SMOOTHING_WINDOW = 7

# 1. Define the shared base directories
base_path = Path("/home/psxkf4/offline_training_phypush/deployed_models")

# 2. Extract the incredibly long experiment folder name as a variable
exp_folder_v1 = "velema1.0std0.0005_realCalibRubW5_b64_lroptAdamW_lrscheReduceLROnPlateau_msharp30.0_musharp10.0_dropout0.4_numepo500_transver4_mseenmax1.0_mseenmin0.3"
exp_folder_v2 = "velema1.0_60_aft0.3_vft0.01_b64_lroptAdamW_lrscheOneCycle_msharp5.0_musharp5.0_dropout0.2_dmodel64_ms20.0_fs1.0_nenc4_numepo1000_transver4_mseenmax1.0_mseenmin0.3"
exp_folder_v3 = "velema1.0_60_aft0.3_vft0.01_b64_lroptAdamW_lrscheOneCycle_msharp5.0_musharp5.0_dropout0.2_dmodel64_ms1.0_fs1.0_nenc4_numepo1000_transver4_mseenmax1.0_mseenmin0.3"
exp_folder_v4 = "velema1.0_60_aft0.3_vft0.01_b64_lroptAdamW_lrscheOneCycle_msharp5.0_musharp5.0_dropout0.4_dmodel64_ms1.0_fs1.0_nenc4_numepo500_transver4_mseenmax1.0_mseenmin0.3"

exp_path = base_path / exp_folder_v4

# 3. Create a Registry mapping Version Tags to their specific Model Paths
MODEL_REGISTRY = {
    "v1_hybrid": exp_path / "hybrid_tcri-log1p_mse_task10.0_pcri-mse_p5c5.0.pth",

    "v1_pinn":   exp_path / "pinn_pcri-mse_p5c10.0.pth",
    "v2_pinn":   exp_path / "pinn_pcri-mse_p10c10.0.pth",
    "v3_pinn":   exp_path / "pinn_pcri-L1_p5c10.0 [Drop0.4_Epo500].pth",
    "v4_pinn":   exp_path / "pinn_pcri-L1_p10c10.0 [Drop0.4_Epo500].pth",
    "v5_pinn":   exp_path / "pinn_annstartepo300_pcri-L1_p10c10.0_p9-2c5.0 [Drop0.4_Epo500].pth",

    "v1_data":   exp_path / "data_tcri-log1p_mse_task10.0.pth",
}

# 4. Explicitly assign the Version Tag you want to actively use
VERSION_TAG = "v3_pinn"  

# Automatically grab the correct path based on the tag
if VERSION_TAG not in MODEL_REGISTRY:
    raise ValueError(f"❌ Invalid VERSION_TAG: '{VERSION_TAG}'. Choose from {list(MODEL_REGISTRY.keys())}")

MODEL_PATH = MODEL_REGISTRY[VERSION_TAG]
print(f"Loaded Target: {VERSION_TAG} -> {MODEL_PATH.name}")


EXPERIMENT_FOLDER = f"mgt_0.57_60_rub_w{SMOOTHING_WINDOW}"
DATA_COLLECTION = 0


# ==========================================
# UTILS
# ==========================================
def ensure_dirs():
    cwd = os.getcwd()
    for folder in ['vis', 'csv_data']:
        if not os.path.exists(os.path.join(cwd, folder)):
            os.makedirs(os.path.join(cwd, folder))
    return cwd

def load_model(device):
    model = PhysicsTransformerEstimator(
        input_dim=1, 
        d_model=64,             # Updated to match training (was 32)
        nhead=4, 
        num_encoder_layers=4,   # Updated to match training (was 2)
        seq_len=60, 
        dropout=0.0,            # Force 0 for inference
        m_sharpness=5.0,        # Updated from training config
        mu_sharpness=5.0,       # Updated from training config
        version=4
    )
    if os.path.exists(MODEL_PATH):
        try:
            # Map to CPU to avoid CUDA version errors
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            model.to(device).eval()
            print(f"✅ Model loaded successfully: {MODEL_PATH}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
        return None

def process_and_inference(model, history_vel, history_acc, device, version_tag, save_dir="."):
    """
    Updates:
    1. Finds the minimum acceleration (negative peak / maximum deceleration).
    2. Extracts Inference Window (60 steps) based on the peak.
    3. Plots the FULL sequence, highlighting the 60 steps.
    4. SAVES A SEPARATE PLOT showing ONLY the 60-step extracted velocity input.
    5. Saves CSV containing strictly a 100-step context window, marking the 60-step inference region.
    """

    # --- Configuration ---
    T_BEFORE = -3
    T_AFTER = 57 
    WINDOW_LEN = T_AFTER - T_BEFORE  
    CTX_LEN = 100  
    PAD_BEFORE_INF = 20  

    # Convert full history to numpy arrays [TotalSteps, 6]
    np_vel_full = np.array(history_vel)
    np_acc_full = np.array(history_acc)
    total_steps = len(np_vel_full)

    if total_steps < CTX_LEN:
        print(f"⚠️ Not enough data (Total Steps: {total_steps}). Need at least {CTX_LEN}.")
        return

    # 1. Find Negative Peak Acceleration Index (Impact Deceleration)
    acc_push_axis = np_acc_full[:, PUSH_AXIS_IDX]
    vel_push_axis = np_vel_full[:, PUSH_AXIS_IDX]
    
    skip_start = 100
    skip_end = 100
    
    # Check if we actually have enough data to skip 200 steps
    if total_steps > (skip_start + skip_end):
        search_window_vel = vel_push_axis[skip_start : -skip_end]
        t_peak_vel = int(np.argmin(search_window_vel)) + skip_start

        search_window_acc = acc_push_axis[skip_start : -skip_end]
        t_peak_acc = int(np.argmin(search_window_acc)) + skip_start

        t_peak_diff = abs(t_peak_acc - t_peak_vel)
        T_BEFORE-=t_peak_diff
        t_peak = t_peak_vel
        print("t_peak_acc: ", t_peak_acc)
        print("t_peak_vel: ", t_peak_vel)
    else:
        print(f"⚠️ Sequence too short ({total_steps} steps) to exclude 100 from both ends. Searching full sequence.")
        t_peak = int(np.argmin(acc_push_axis))
        
    print(f"--- Detected Impact (Negative Peak) at Step {t_peak} ---")

    # 2. Define Inference Indices (Absolute)
    inf_start_abs = max(0, t_peak + T_BEFORE)
    inf_end_abs = inf_start_abs + WINDOW_LEN
    
    if inf_end_abs > total_steps:
        print(f"⚠️ Peak is too close to end of recording. Shifting window back.")
        shift = inf_end_abs - total_steps
        inf_start_abs -= shift
        inf_end_abs -= shift

    # 3. Extract 60-step Data (for Model & Plotting)
    vel_60 = np_vel_full[inf_start_abs:inf_end_abs]
    acc_60 = np_acc_full[inf_start_abs:inf_end_abs]

    # 4. Define 100-step Context Window (Specifically for CSV)
    ctx_start_abs = max(0, inf_start_abs - PAD_BEFORE_INF)
    ctx_end_abs = ctx_start_abs + CTX_LEN
    
    if ctx_end_abs > total_steps:
        ctx_end_abs = total_steps
        ctx_start_abs = max(0, ctx_end_abs - CTX_LEN)

    vel_100 = np_vel_full[ctx_start_abs:ctx_end_abs]
    acc_100 = np_acc_full[ctx_start_abs:ctx_end_abs]

    csv_inf_start = inf_start_abs - ctx_start_abs
    csv_inf_end = inf_end_abs - ctx_start_abs

    # --- INFERENCE (Optional) ---
    mass_est = 0.0
    mu_est = 0.0

    if model is not None:
        if len(vel_60) != WINDOW_LEN:
            print(f"⚠️ Window size mismatch: Got {len(vel_60)}, expected {WINDOW_LEN}. Skipping inference.")
        else:
            t_vel = torch.from_numpy(vel_60[:, PUSH_AXIS_IDX]).float().view(1, WINDOW_LEN, 1).to(device)

            with torch.no_grad():
                output = model(t_vel)
                mass_est = output[0, 0].item()
                mu_est = output[0, 1].item()
            
            print("\n" + "="*40)
            print("    PREDICTION RESULTS ")
            print("="*40)
            print(f"  MASS: {mass_est:.4f} kg")
            print(f"  MU:   {mu_est:.4f}")
            print("="*40 + "\n")
    else:
        print("Model not provided. Skipping inference, proceeding to save data.")

    # --- PREPARE DIRECTORIES ---
    experiment_folder = EXPERIMENT_FOLDER
    if DATA_COLLECTION:
        category = "data_collection"
    else:
        category = "inference"
    
    vis_dir = os.path.join(save_dir, "vis", category, experiment_folder)
    csv_dir = os.path.join(save_dir, "csv_data", category, experiment_folder)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S") 
    
    # ---> NEW: Embedded `version_tag` into the filenames <---
    if DATA_COLLECTION:
        base_filename = f"ext_{WINDOW_LEN}steps_{timestamp}_w{SMOOTHING_WINDOW}_{version_tag}"
    else:
        base_filename = f"ext_{WINDOW_LEN}steps_mest{mass_est:.3f}_muest{mu_est:.3f}_w{SMOOTHING_WINDOW}_{version_tag}"

    # ==========================================
    # --- PLOT 1: FULL SEQUENCE OVERVIEW ---
    # ==========================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Also added the version tag to the plot title for clarity!
    fig.suptitle(f"[{version_tag.upper()}] Full Sequence ({total_steps} steps) | Extracted: {WINDOW_LEN} | Pred: Mass = {mass_est:.4f} kg, Mu = {mu_est:.4f}", 
                 fontsize=14, fontweight='bold')

    steps_full = np.arange(total_steps)
    steps_60 = np.arange(inf_start_abs, inf_end_abs)

    axes[0].plot(steps_full, np_vel_full[:, PUSH_AXIS_IDX], 'b-', alpha=0.3, label=f'Full Sequence ({total_steps})')
    axes[0].plot(steps_60, vel_60[:, PUSH_AXIS_IDX], 'b-', linewidth=3, label=f'Extracted Window ({WINDOW_LEN})')
    axes[0].set_title(f"Velocity: Highlighted Window (Steps {inf_start_abs} to {inf_end_abs-1})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps_full, np_acc_full[:, PUSH_AXIS_IDX], 'r-', alpha=0.3, label=f'Full Sequence ({total_steps})')
    axes[1].plot(steps_60, acc_60[:, PUSH_AXIS_IDX], 'r-', linewidth=3, label=f'Extracted Window ({WINDOW_LEN})')
    axes[1].axvline(t_peak, color='k', linestyle='--', alpha=0.5, label='Impact Peak (Min)')
    axes[1].set_title(f"Acceleration: Min Peak at {t_peak}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    
    plot_path_full = os.path.join(vis_dir, f"{base_filename}.png")
    plt.savefig(plot_path_full)
    plt.close()

    # ==========================================
    # --- PLOT 2: EXTRACTED VELOCITY ONLY ---
    # ==========================================
    fig_vel, ax_vel = plt.subplots(figsize=(8, 4))
    
    ax_vel.plot(np.arange(WINDOW_LEN), vel_60[:, PUSH_AXIS_IDX], 'b-o', linewidth=2, markersize=4)
    ax_vel.set_title(f"Model Input: Extracted Velocity ({WINDOW_LEN} steps) [{version_tag}]")
    ax_vel.set_xlabel("Local Time Step")
    ax_vel.set_ylabel(f"Velocity (Axis {PUSH_AXIS_IDX}) [m/s]")
    ax_vel.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path_vel = os.path.join(vis_dir, f"{base_filename}_input_vel.png")
    plt.savefig(plot_path_vel)
    plt.close()

    # ==========================================
    # --- SAVE CSV (Using 100-step Context) ---
    # ==========================================
    csv_path = os.path.join(csv_dir, f"{base_filename}.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step_local', 'v_y', 'a_y', 'is_inference_region'])
        for i in range(len(vel_100)):
            is_inf = 1 if (i >= csv_inf_start and i < csv_inf_end) else 0
            writer.writerow([i, vel_100[i, PUSH_AXIS_IDX], acc_100[i, PUSH_AXIS_IDX], is_inf])
    
    print(f"Full Plot saved to {plot_path_full}")
    print(f"Input Vel Plot saved to {plot_path_vel}")
    print(f"Data saved to {csv_path}")

# ==========================================
# MAIN LOOP
# ==========================================
def run_push_and_velocity():
    logging.basicConfig(level=logging.INFO)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("CUDA unavailable. Using CPU.")
        device = torch.device('cpu')

    model = load_model(device)
    
    history_vel = []
    history_acc = []
    vel_smoothing_buffer = []
    prev_vel_w = np.zeros(6)

    try:
        print(f"Connecting to {hostname}...")
        panda = panda_py.Panda(hostname)
        gripper = libfranka.Gripper(hostname)
        
        print(f"Moving to Pushset Pose...")
        panda.move_to_joint_position(PUSHSET_Q, speed_factor=0.2)
        time.sleep(2.0)
        
        # --- Velocity Control ---
        print(f"Starting Base Velocity Control: {V_DESIRED_BASE}")
        
        ctrl = controllers.IntegratedVelocity()
        panda.start_controller(ctrl)
        model_robot = panda.get_model()
        
        with panda.create_context(frequency=1/DT) as ctx:
            start_time = time.time()
            
            while ctx.ok():
                if time.time() - start_time > VELOCITY_DURATION:
                    break
                
                state = panda.get_state()
                J_flat = model_robot.zero_jacobian(libfranka.Frame.kEndEffector, state)
                J = np.array(J_flat).reshape((6, 7), order='F')
                J_pinv = np.linalg.pinv(J)
                
                # Command
                dq_cmd = J_pinv @ V_DESIRED_BASE
                ctrl.set_control(dq_cmd)
                
                # Record
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

        print("Motion finished.")
        panda.stop_controller()
        
        # ---> NEW: Passed `VERSION_TAG` into processing <---
        process_and_inference(model, history_vel, history_acc, device, VERSION_TAG)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_push_and_velocity()