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

V_DESIRED_BASE_SET = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]) 
V_DESIRED_BASE_PUSH = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0]) # Move 8 cm/s 
PUSH_AXIS_IDX = 1  # 0=X, 1=Y, 2=Z (Must match non-zero element in V_DESIRED)

# Duration parameters
SET_VELOCITY_DURATION = 3.0 
PUSH_VELOCITY_DURATION = 7.0 
DT = 0.01  # Fixed time step (1/100 Hz)


# 1. Define the shared base directories
base_path = Path("/home/psxkf4/offline_training_phypush/deployed_models")

# 2. Extract the incredibly long experiment folder name as a variable
exp_folder_v1 = "velema1.0std0.0005_realCalibRubW5_b64_lroptAdamW_lrscheReduceLROnPlateau_msharp30.0_musharp10.0_dropout0.4_numepo500_transver4_mseenmax1.0_mseenmin0.3"
exp_folder_v2 = "velema1.0_60_aft0.3_vft0.01_b64_lroptAdamW_lrscheOneCycle_msharp5.0_musharp5.0_dropout0.2_dmodel64_ms20.0_fs1.0_nenc4_numepo1000_transver4_mseenmax1.0_mseenmin0.3"
exp_folder_v3 = "velema1.0_60_aft0.3_vft0.01_b64_lroptAdamW_lrscheOneCycle_msharp5.0_musharp5.0_dropout0.2_dmodel64_ms1.0_fs1.0_nenc4_numepo1000_transver4_mseenmax1.0_mseenmin0.3"
exp_folder_v4 = "velema1.0_60_aft0.3_vft0.01_b64_lroptAdamW_lrscheOneCycle_msharp5.0_musharp5.0_dropout0.4_dmodel64_ms1.0_fs1.0_nenc4_numepo500_transver4_mseenmax1.0_mseenmin0.3"
exp_folder_v5 = "velema1.0_60_aft0.3_vft0.01_b64_lroptAdamW_lrscheOneCycle_msharp5.0_musharp5.0_dropout0.4_dmodel64_ms1.0_fs10.0_nenc4_numepo1000_transver4_mseenmax1.0_mseenmin0.3"
exp_folder_v6 = "velema1.0_60_aft0.3_vft0.01_b64_lroptAdamW_lrscheOneCycle_msharp5.0_musharp5.0_dropout0.4_dmodel64_ms1.0_fs1.0_nenc4_numepo1000_transver4_mseenmax1.0_mseenmin0.3"

exp_path = base_path / exp_folder_v6

# 3. Create a Registry mapping Version Tags to their specific Model Paths
MODEL_REGISTRY = {
    "v1_hybrid": exp_path / "hybrid_tcri-log1p_mse_task10.0_pcri-mse_p5c5.0.pth",

    "v1_pinn":   exp_path / "pinn_pcri-mse_p5c10.0.pth",
    "v2_pinn":   exp_path / "pinn_pcri-mse_p10c10.0.pth",
    "v3_pinn":   exp_path / "pinn_pcri-L1_p5c10.0 [Drop0.4_Epo500].pth",
    "v4_pinn":   exp_path / "pinn_pcri-L1_p10c10.0 [Drop0.4_Epo500].pth",
    "v5_pinn":   exp_path / "pinn_annstartepo300_pcri-L1_p10c10.0_p9-2c5.0 [Drop0.4_Epo500].pth",
    "v6_pinn":   exp_path / "pinn_pcri-L1_p10c10.0 [FS10_Drop0.4_Epo1000].pth",

    "v1_data":   exp_path / "data_tcri-log1p_mse_task10.0 [Drop0.4_Epo1000].pth",
}


SMOOTHING_WINDOW = 7
VERSION_TAG = "v1_data"  
M_GT = 0.39
MU_GT = None
EXPERIMENT_FOLDER = f"mgt{M_GT}_w{SMOOTHING_WINDOW}"
OBJECT = "nolid_cube"
SURFACE = "green_rub"
DATA_COLLECTION = 0

MASS_RANGE = 1.9  # m_unseen_max (2.0) - m_unseen_min (0.1)
MU_RANGE = 0.4    # mu_unseen_max (0.6) - mu_unseen_min (0.2)

# Automatically grab the correct path based on the tag
if VERSION_TAG not in MODEL_REGISTRY:
    raise ValueError(f"❌ Invalid VERSION_TAG: '{VERSION_TAG}'. Choose from {list(MODEL_REGISTRY.keys())}")

MODEL_PATH = MODEL_REGISTRY[VERSION_TAG]
print(f"Loaded Target: {VERSION_TAG} -> {MODEL_PATH.name}")




# ==========================================
# UTILS
# ==========================================
def calculate_metrics(gt, est, range_val):
    """Helper to compute nMAE, NRMSE, and sMAPE."""
    gt = np.array(gt).flatten()
    est = np.array(est).flatten()
    
    # nMAE (%)
    mae = np.mean(np.abs(est - gt))
    nmae_pct = (mae / range_val) * 100 if range_val > 0 else 0
    
    # NRMSE (%)
    rmse = np.sqrt(np.mean((est - gt)**2))
    nrmse_pct = (rmse / range_val) * 100 if range_val > 0 else 0
    
    # sMAPE (%) - Bounded between 0 and 200
    # Formula: (100 / n) * sum(|est - gt| / ((|gt| + |est|) / 2))
    denominator = (np.abs(gt) + np.abs(est)) / 2
    # Avoid division by zero for cases where both gt and est are 0
    smape_pct = np.mean(np.abs(est - gt) / np.maximum(denominator, 1e-8)) * 100
    
    return {
        "mae": float(mae), 
        "nmae_pct": float(nmae_pct), 
        "nrmse_pct": float(nrmse_pct), 
        "smape_pct": float(smape_pct)
    }

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
    
    if total_steps > (skip_start + skip_end):
        # A. First find the global minimum for velocity in the safe middle region
        search_window_vel = vel_push_axis[skip_start : -skip_end]
        t_peak_vel = int(np.argmin(search_window_vel)) + skip_start

        # B. Define a local search window (e.g., 20 steps before and after the velocity drop)
        acc_search_radius = 20
        local_start = max(0, t_peak_vel - acc_search_radius)
        local_end = min(total_steps, t_peak_vel + acc_search_radius)

        # C. Find the maximum deceleration (min acceleration) strictly within this local window
        local_search_window_acc = acc_push_axis[local_start:local_end]
        t_peak_acc = int(np.argmin(local_search_window_acc)) + local_start

        # D. Set the official peak to the exact moment of impact (Acceleration Peak)
        t_peak = t_peak_acc

        print(f"t_peak_vel (Velocity Drop): {t_peak_vel}")
        print(f"t_peak_acc (Impact Spike):  {t_peak_acc}")
        print(f"Time Difference:            {abs(t_peak_acc - t_peak_vel)} steps")
        
    else:
        print(f"⚠️ Sequence too short ({total_steps} steps). Searching full sequence.")
        t_peak = int(np.argmin(acc_push_axis))
        
    print(f"--- Final Anchored Impact Peak at Step {t_peak} ---")

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

   # --- INFERENCE ---
    mass_est = 0.0
    mu_est = 0.0
    
    # ---> FIX: Initialize default metrics here so they always exist! <---
    mass_metrics = {"mae": "N/A", "nmae_pct": "N/A", "nrmse_pct": "N/A", "smape_pct": "N/A"}
    mu_metrics = {"mae": "N/A", "nmae_pct": "N/A", "nrmse_pct": "N/A", "smape_pct": "N/A"}

    if model is not None:
        if len(vel_60) != WINDOW_LEN:
            print(f"⚠️ Window size mismatch: Got {len(vel_60)}, expected {WINDOW_LEN}. Skipping inference.")
        else:
            t_vel = torch.from_numpy(vel_60[:, PUSH_AXIS_IDX]).float().view(1, WINDOW_LEN, 1).to(device)

            with torch.no_grad():
                output = model(t_vel)
                mass_est = output[0, 0].item()
                mu_est = output[0, 1].item()
            
            # Calculate statistical metrics if GT exists
            if M_GT is not None:
                mass_metrics = calculate_metrics(M_GT, mass_est, MASS_RANGE)
            if MU_GT is not None:
                mu_metrics = calculate_metrics(MU_GT, mu_est, MU_RANGE)
            
            print("\n" + "="*50)
            print("                🔮 PREDICTION RESULTS 🔮")
            print("="*50)
            print(f"  MASS PRED: {mass_est:.4f} kg   |   GT: {M_GT if M_GT else 'N/A'}")
            if M_GT is not None:
                print(f"   -> nMAE: {mass_metrics['nmae_pct']:.2f}% | NRMSE: {mass_metrics['nrmse_pct']:.2f}% | sMAPE: {mass_metrics['smape_pct']:.2f}%")
            
            print("-" * 50)
            print(f"  MU PRED:   {mu_est:.4f}      |   GT: {MU_GT if MU_GT else 'N/A'}")
            if MU_GT is not None:
                print(f"   -> nMAE: {mu_metrics['nmae_pct']:.2f}% | NRMSE: {mu_metrics['nrmse_pct']:.2f}% | sMAPE: {mu_metrics['smape_pct']:.2f}%")
            print("="*50 + "\n")
    else:
        print("Model not provided. Skipping inference, proceeding to save data.")

    # --- PREPARE DIRECTORIES ---
    experiment_folder = EXPERIMENT_FOLDER
    condition = f"{OBJECT}_{SURFACE}"
    if DATA_COLLECTION:
        purpose = "data_collection"
    else:
        purpose = "inference"
    
    vis_dir = os.path.join(save_dir, "vis", purpose, condition, experiment_folder, version_tag)
    csv_dir = os.path.join(save_dir, "csv_data", purpose, condition, experiment_folder, version_tag)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S") 
    base_filename = f"ext_mest{mass_est:.3f}_muest{mu_est:.3f}"

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
    
    # ==========================================
    # --- SAVE METADATA CSV ---
    # ==========================================
    # Ordered to match the directory structure: 
    # Purpose -> Condition -> Experiment Folder -> Version Tag
    metadata = {
        # --- Hierarchy Level 1 ---
        "PURPOSE": purpose,
        "DATA_COLLECTION": DATA_COLLECTION,
        
        # --- Hierarchy Level 2 ---
        "CONDITION": condition,
        "OBJECT": OBJECT,
        "SURFACE": SURFACE,
        
        # --- Hierarchy Level 3 ---
        "EXPERIMENT_FOLDER": experiment_folder,
        "SMOOTHING_WINDOW": SMOOTHING_WINDOW,
        
        # --- Hierarchy Level 4 ---
        "VERSION_TAG": version_tag,
        
        # --- Execution Details & Predictions ---
        "TIMESTAMP": timestamp,
        "WINDOW_LEN": WINDOW_LEN,
        "GLOBAL_IMPACT_STEP": t_peak,
        "PRED_MASS_KG": mass_est,
        "MGT": M_GT if M_GT else "N/A",
        "PRED_MU": mu_est,
        "MU_GT": MU_GT if MU_GT else "N/A",
        
        # --- Mass Metrics ---
        "MASS_MAE": mass_metrics["mae"],
        "MASS_NMAE_PCT": mass_metrics["nmae_pct"],
        "MASS_NRMSE_PCT": mass_metrics["nrmse_pct"],
        "MASS_SMAPE_PCT": mass_metrics["smape_pct"],
        
        # --- Mu Metrics ---
        "MU_MAE": mu_metrics["mae"],
        "MU_NMAE_PCT": mu_metrics["nmae_pct"],
        "MU_NRMSE_PCT": mu_metrics["nrmse_pct"],
        "MU_SMAPE_PCT": mu_metrics["smape_pct"],
    }

    metadata_csv_path = os.path.join(csv_dir, f"{base_filename}_metadata.csv")
    
    with open(metadata_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])  # Header
        for key, value in metadata.items():
            writer.writerow([key, value])
            

    print(f"Full Plot saved to {plot_path_full}")
    print(f"Input Vel Plot saved to {plot_path_vel}")
    print(f"Data saved to {csv_path}")
    print(f"Metadata saved to {metadata_csv_path}")

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

        with panda.create_context(frequency=1/DT) as ctx_set:
            start_time_set = time.time()
            
            while ctx_set.ok():
                if time.time() - start_time_set > SET_VELOCITY_DURATION:
                    break
                
                state = panda.get_state()
                J_flat = model_robot.zero_jacobian(libfranka.Frame.kEndEffector, state)
                J = np.array(J_flat).reshape((6, 7), order='F')
                J_pinv = np.linalg.pinv(J)
                
                # Command
                dq_cmd = J_pinv @ V_DESIRED_BASE_SET
                ctrl.set_control(dq_cmd)
                
        # ---> FIX: STOP the controller before doing a joint motion! <---
        panda.stop_controller()
        time.sleep(3.0)

        # 2. Now it is safe to use built-in joint position commands
        print(f"Moving to Pushset Pose...")
        panda.move_to_joint_position(PUSHSET_Q, speed_factor=0.2)
        time.sleep(0.5)
        
        # ---> RESTART the velocity controller for the push! <---
        panda.start_controller(ctrl)
        
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
                
        # Final cleanup
        panda.stop_controller()
        
        process_and_inference(model, history_vel, history_acc, device, VERSION_TAG)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_push_and_velocity()