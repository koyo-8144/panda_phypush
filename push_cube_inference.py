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

SMOOTHING_WINDOW = 10

# 1. Define the shared base directories
base_path = Path("/home/psxkf4/offline_training_phypush/deployed_models")
# 2. Extract the incredibly long experiment folder name as a variable
exp_folder = "velema1.0std0.0005_realCalibRubW5_b64_lroptAdamW_lrscheReduceLROnPlateau_msharp30.0_musharp10.0_dropout0.4_numepo500_transver4_mseenmax1.0_mseenmin0.3"
exp_path = base_path / exp_folder
# 3. Store both models distinctly so they don't overwrite each other
HYBRID_MODEL_PATH = exp_path / "hybrid_tcri-log1p_mse_task10.0_pcri-mse_p5c5.0.pth"
PINN_MODEL_PATH = exp_path / "pinn_pcri-mse_p5c10.0.pth"
# PINN_MODEL_PATH = exp_path / "pinn_pcri-mse_p10c10.0.pth"
DATA_MODEL_PATH = exp_path /"data_tcri-log1p_mse_task10.0.pth"
# 4. Explicitly assign the one you want to actively use
MODEL_PATH = PINN_MODEL_PATH


EXPERIMENT_FOLDER = f"mgt_0.76_60_rub_w{SMOOTHING_WINDOW}"


# ==========================================
# 2. TRANSFORMER MODEL ARCHITECTURE
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PhysicsTransformerEstimator(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, num_encoder_layers=2, seq_len=60, dropout=0.4, version=4):
        super().__init__()
        self.version = version
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.net_force_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)) 
        self.fric_force_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)) 

        # Missing layers from checkpoint
        self.phys_net_proj = nn.Linear(d_model, 1)
        self.phys_fric_proj = nn.Linear(d_model, 1)

        self.q_mass = nn.Parameter(torch.randn(1, 1, d_model) * 2.0)
        self.mass_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
        self.mass_pred_mlp = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

        self.q_fric = nn.Parameter(torch.randn(1, 1, d_model) * 2.0)
        self.fric_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
        self.mu_pred_mlp = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, x_vel):
        x = x_vel
        z = self.input_proj(x)
        z = self.pos_encoder(z)
        h_enc = self.transformer_encoder(z)

        feat_net = self.net_force_mlp(h_enc)
        feat_fric = self.fric_force_mlp(h_enc)

        q_m = self.q_mass.expand(x.size(0), -1, -1)
        mass_ctx, _ = self.mass_attn(query=q_m, key=feat_net, value=feat_net)
        mass_pred = self.mass_pred_mlp(mass_ctx.squeeze(1))

        q_f = self.q_fric.expand(x.size(0), -1, -1)
        fric_ctx, _ = self.fric_attn(query=q_f, key=feat_fric, value=feat_fric)
        mu_pred = self.mu_pred_mlp(fric_ctx.squeeze(1))

        return torch.cat([mass_pred, mu_pred], dim=-1)



# ==========================================
# 3. UTILS
# ==========================================
def ensure_dirs():
    cwd = os.getcwd()
    for folder in ['vis', 'csv_data']:
        if not os.path.exists(os.path.join(cwd, folder)):
            os.makedirs(os.path.join(cwd, folder))
    return cwd

def load_model(device):
    model = PhysicsTransformerEstimator(input_dim=1, d_model=32, seq_len=60, dropout=0.0, version=4)
    if os.path.exists(MODEL_PATH):
        try:
            # Map to CPU to avoid CUDA version errors
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            model.to(device).eval()
            print(f"✅ Model loaded: {MODEL_PATH}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
        return None


def process_and_inference(model, history_vel, history_acc, device, save_dir="."):
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
    WINDOW_LEN = T_AFTER - T_BEFORE  # Exactly 60 steps
    CTX_LEN = 100  # Total context length for the CSV
    PAD_BEFORE_INF = 20  # Steps to show in CSV before the 60-step window starts

    # Convert full history to numpy arrays [TotalSteps, 6]
    np_vel_full = np.array(history_vel)
    np_acc_full = np.array(history_acc)
    total_steps = len(np_vel_full)

    if total_steps < CTX_LEN:
        print(f"⚠️ Not enough data (Total Steps: {total_steps}). Need at least {CTX_LEN}.")
        return

    # 1. Find Negative Peak Acceleration Index (Impact Deceleration)
    acc_push_axis = np_acc_full[:, PUSH_AXIS_IDX]
    t_peak = int(np.argmin(acc_push_axis))
    
    print(f"--- Detected Impact (Negative Peak) at Step {t_peak} ---")

    # 2. Define Inference Indices (Absolute)
    inf_start_abs = max(0, t_peak + T_BEFORE)
    inf_end_abs = inf_start_abs + WINDOW_LEN
    
    # Safety Check: Ensure we don't go out of bounds at the end
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
    
    # Boundary check for context
    if ctx_end_abs > total_steps:
        ctx_end_abs = total_steps
        ctx_start_abs = max(0, ctx_end_abs - CTX_LEN)

    # Extract 100-step Context Data
    vel_100 = np_vel_full[ctx_start_abs:ctx_end_abs]
    acc_100 = np_acc_full[ctx_start_abs:ctx_end_abs]

    # Calculate local indices for the CSV `is_inference_region` flag
    csv_inf_start = inf_start_abs - ctx_start_abs
    csv_inf_end = inf_end_abs - ctx_start_abs

    # --- INFERENCE (Optional) ---
    mass_est = 0.0
    mu_est = 0.0

    if model is not None:
        if len(vel_60) != WINDOW_LEN:
            print(f"⚠️ Window size mismatch: Got {len(vel_60)}, expected {WINDOW_LEN}. Skipping inference.")
        else:
            # Reshape to [Batch=1, Seq=60, Dim=1]
            t_vel = torch.from_numpy(vel_60[:, PUSH_AXIS_IDX]).float().view(1, WINDOW_LEN, 1).to(device)
            # t_acc = torch.from_numpy(acc_60[:, PUSH_AXIS_IDX]).float().view(1, WINDOW_LEN, 1).to(device)

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
    
    vis_dir = os.path.join(save_dir, "vis", experiment_folder)
    csv_dir = os.path.join(save_dir, "csv_data", experiment_folder)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S") # Generate unique timestamp
    base_filename = f"ext_{WINDOW_LEN}steps_{timestamp}_w{SMOOTHING_WINDOW}"

    # ==========================================
    # --- PLOT 1: FULL SEQUENCE OVERVIEW ---
    # ==========================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    fig.suptitle(f"Full Sequence ({total_steps} steps) | Extracted: {WINDOW_LEN} | Pred: Mass = {mass_est:.4f} kg, Mu = {mu_est:.4f}", 
                 fontsize=14, fontweight='bold')

    steps_full = np.arange(total_steps)
    steps_60 = np.arange(inf_start_abs, inf_end_abs)

    # Plot Velocity
    axes[0].plot(steps_full, np_vel_full[:, PUSH_AXIS_IDX], 'b-', alpha=0.3, label=f'Full Sequence ({total_steps})')
    axes[0].plot(steps_60, vel_60[:, PUSH_AXIS_IDX], 'b-', linewidth=3, label=f'Extracted Window ({WINDOW_LEN})')
    axes[0].set_title(f"Velocity: Highlighted Window (Steps {inf_start_abs} to {inf_end_abs-1})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot Acceleration
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
    ax_vel.set_title(f"Model Input: Extracted Velocity ({WINDOW_LEN} steps)")
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
# 4. MAIN LOOP
# ==========================================
def run_push_and_velocity():
    logging.basicConfig(level=logging.INFO)
    
    # Check CUDA
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
        
        # # --- Init ---
        # print("Moving to Neutral...")
        # panda.move_to_start()
        
        # print("Closing Gripper...")
        # gripper.grasp(0, 0.2, 10, 0.04, 0.04)
        # time.sleep(1.0)
        
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
            count = 0
            
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
                
                # No rotation needed for Base Frame recording (Base=World)
                history_vel.append(vel_w_smoothed)
                history_acc.append(acc_w)

                # count+=1
                # print("count: ", count)


        print("Motion finished.")
        panda.stop_controller()
        
        # print("Cleanup...")
        # panda.move_to_start()
        # gripper.move(0.08, 0.2)
        
        # --- Processing ---
        process_and_inference(model, history_vel, history_acc, device)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_push_and_velocity()