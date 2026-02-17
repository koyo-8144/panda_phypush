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

# ==========================================
# 1. CONFIGURATION
# ==========================================
hostname = '172.22.2.3'
username = 'cobotmakerspace'
password = 'cobotmakerspace'

# COMMAND: Move 8 cm/s along Base Y-Axis (Index 1)
V_DESIRED_BASE = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0]) 
PUSH_AXIS_IDX = 1  # 0=X, 1=Y, 2=Z (Must match non-zero element in V_DESIRED)

# Duration parameters
VELOCITY_DURATION = 3.0 
DT = 0.01  # Fixed time step (1/100 Hz)

# The specific joint configuration (radians)
PUSHSET_Q = np.array([
    1.72169, -1.02605, -2.27493, -2.10522, 0.503725, 1.85432, -1.62299
])

SMOOTHING_WINDOW = 5 
MODEL_PATH = 'trained_models/transformer_epoch500.pth'

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
    def __init__(self, input_dim=2, d_model=32, nhead=4, num_encoder_layers=2, seq_len=20, dropout=0.4, version=4):
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

    def forward(self, x_acc, x_vel):
        x = torch.cat([x_vel, x_acc], dim=-1) 
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
    model = PhysicsTransformerEstimator(input_dim=2, d_model=32, seq_len=20, version=4)
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

def process_and_inference(model, history_vel, history_acc, device):
    """
    Updates:
    1. Finds peak acceleration (t_peak).
    2. Extracts Inference Window using Reference Logic:
       - Start: t_peak + t_before (15)
       - End:   t_peak + t_after  (35)
       - Len:   20 steps
    3. Extracts a broader 100-step context for visualization/debugging.
    """
    # --- Configuration from Reference Code ---
    T_BEFORE = 15
    T_AFTER = 35
    WINDOW_LEN = T_AFTER - T_BEFORE  # 20 steps

    if len(history_vel) < 100:
        print(f"⚠️ Not enough data (Steps: {len(history_vel)}). Need > 100.")
        return

    # Convert full history to numpy arrays [TotalSteps, 6]
    np_vel_full = np.array(history_vel)
    np_acc_full = np.array(history_acc)

    # 1. Find Peak Acceleration Index on the Pushing Axis
    # Base Y = Index 1 (Matches your V_DESIRED [0, 0.08, 0])
    acc_push_axis = np_acc_full[:, PUSH_AXIS_IDX]
    t_peak = np.argmax(np.abs(acc_push_axis))
    
    print(f"--- Detected Impact Peak at Step {t_peak} ---")

    # 2. Define Inference Indices (Absolute)
    # Reference Logic: start = peak + t_before
    inf_start_abs = t_peak + T_BEFORE
    inf_end_abs = t_peak + T_AFTER
    
    # Safety Check: Ensure we don't go out of bounds
    if inf_end_abs > len(np_vel_full):
        print(f"⚠️ Peak is too close to end of recording. (Peak: {t_peak}, Need: {inf_end_abs})")
        # Fallback: Shift window back to fit
        shift = inf_end_abs - len(np_vel_full)
        inf_start_abs -= shift
        inf_end_abs -= shift

    # 3. Define 100-Step Context Window (Relative to Inference Window)
    # We want the context to surround the inference window comfortably.
    # Let's start the context 40 steps before the inference window starts.
    ctx_start_abs = max(0, inf_start_abs - 40)
    ctx_end_abs = ctx_start_abs + 100
    
    # Boundary check for context
    if ctx_end_abs > len(np_vel_full):
        ctx_end_abs = len(np_vel_full)
        ctx_start_abs = max(0, ctx_end_abs - 100)

    # 4. Extract Data
    # 100-step Context
    vel_100 = np_vel_full[ctx_start_abs:ctx_end_abs]
    acc_100 = np_acc_full[ctx_start_abs:ctx_end_abs]

    # 20-step Inference (Extracted directly from full history for precision)
    vel_20 = np_vel_full[inf_start_abs:inf_end_abs]
    acc_20 = np_acc_full[inf_start_abs:inf_end_abs]

    # Calculate local indices for visualization highlighting
    # Where does the inference window start *inside* the 100-step array?
    viz_inf_start = inf_start_abs - ctx_start_abs
    viz_inf_end = inf_end_abs - ctx_start_abs

    # --- SAVE CSV (100 Steps) ---
    cwd = ensure_dirs()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(cwd, "csv_data", f"inference_context_{timestamp}.csv")
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step_local', 'v_y', 'a_y', 'is_inference_region'])
        for i in range(len(vel_100)):
            is_inf = 1 if (i >= viz_inf_start and i < viz_inf_end) else 0
            writer.writerow([i, vel_100[i, PUSH_AXIS_IDX], acc_100[i, PUSH_AXIS_IDX], is_inf])
    
    # --- PLOT ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    steps_100 = np.arange(len(vel_100))
    steps_20 = np.arange(viz_inf_start, viz_inf_end)

    # Plot Velocity Y
    axes[0].plot(steps_100, vel_100[:, PUSH_AXIS_IDX], 'b-', alpha=0.3, label='Context (100)')
    axes[0].plot(steps_20, vel_20[:, PUSH_AXIS_IDX], 'b-', linewidth=3, label='Model Input (20)')
    axes[0].set_title(f"Velocity Y: Highlighted Inference Window (Peak+{T_BEFORE} to Peak+{T_AFTER})")
    axes[0].legend()

    # Plot Acceleration Y
    axes[1].plot(steps_100, acc_100[:, PUSH_AXIS_IDX], 'r-', alpha=0.3, label='Context (100)')
    axes[1].plot(steps_20, acc_20[:, PUSH_AXIS_IDX], 'r-', linewidth=3, label='Model Input (20)')
    
    # Mark the peak location relative to this plot
    peak_local = t_peak - ctx_start_abs
    if 0 <= peak_local < len(vel_100):
        axes[1].axvline(peak_local, color='k', linestyle='--', alpha=0.5, label='Impact Peak')

    axes[1].set_title(f"Acceleration Y: Peak at {peak_local}, Window starts +{T_BEFORE}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(cwd, "vis", f"inference_highlight_{timestamp}.png"))
    plt.close()

    # --- INFERENCE ---
    if model is not None:
        if len(vel_20) != 20:
            print(f"⚠️ Window size mismatch: Got {len(vel_20)}, expected 20. Skipping inference.")
            return

        # Reshape to [Batch=1, Seq=20, Dim=1]
        t_vel = torch.from_numpy(vel_20[:, PUSH_AXIS_IDX]).float().view(1, 20, 1).to(device)
        t_acc = torch.from_numpy(acc_20[:, PUSH_AXIS_IDX]).float().view(1, 20, 1).to(device)

        with torch.no_grad():
            output = model(t_acc, t_vel)
        
        print("\n" + "="*40)
        print(f"  PREDICTED MASS: {output[0, 0].item():.4f} kg")
        print(f"  PREDICTED MU:   {output[0, 1].item():.4f}")
        print("="*40 + "\n")
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
        
        # --- Init ---
        print("Moving to Neutral...")
        panda.move_to_start()
        
        print("Closing Gripper...")
        gripper.grasp(0, 0.2, 10, 0.04, 0.04)
        time.sleep(1.0)
        
        print(f"Moving to Pushset Pose...")
        panda.move_to_joint_position(PUSHSET_Q, speed_factor=0.2)
        time.sleep(2.0)
        
        # --- Velocity Control ---
        print(f"Starting Base Velocity Control: {V_DESIRED_BASE}")
        
        ctrl = controllers.IntegratedVelocity()
        panda.start_controller(ctrl)
        model_robot = panda.get_model()
        
        with panda.create_context(frequency=100.0) as ctx:
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
                
                # No rotation needed for Base Frame recording (Base=World)
                history_vel.append(vel_w_smoothed)
                history_acc.append(acc_w)

        print("Motion finished.")
        panda.stop_controller()
        
        print("Cleanup...")
        panda.move_to_start()
        gripper.move(0.08, 0.2)
        
        # --- Processing ---
        process_and_inference(model, history_vel, history_acc, device)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_push_and_velocity()