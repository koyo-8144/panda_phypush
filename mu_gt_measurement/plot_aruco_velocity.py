import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

MU_STATIC_LIMIT = 0.4  # Your static friction reference

def main():
    # 1. Setup Directories
    csv_folder = "aruco_velocity"
    plots_folder = "plots"
    os.makedirs(plots_folder, exist_ok=True)
    
    list_of_files = glob.glob(f"{csv_folder}/*.csv")
    if not list_of_files:
        print("❌ No CSV files found.")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"📊 Visualizing Analysis for: {latest_file}")
    save_path = os.path.join(plots_folder, os.path.basename(latest_file).replace('.csv', '.png'))

    # 2. Load Data and Calculate mu_d for Watermark
    df = pd.read_csv(latest_file)
    df['Relative_Time'] = df['Timestamp'] - df['Timestamp'].iloc[0]

    # Endpoint Logic Synchronization
    df['jump_x'] = df['Px'].diff().abs()
    jump_idx = df['jump_x'].idxmax()
    start_idx = max(0, jump_idx - 1)
    end_idx = jump_idx
    
    dt = df.loc[end_idx, 'Timestamp'] - df.loc[start_idx, 'Timestamp']
    p1 = df.loc[start_idx, ['Px', 'Pz']].values
    p2 = df.loc[end_idx, ['Px', 'Pz']].values
    d = np.linalg.norm(p2 - p1)
    v0 = (2 * d) / dt if dt > 0 else 0
    mu_d = (v0**2) / (2 * 9.81 * d) if d > 0 else 0

    # 3. Create the Dashboard
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    (ax1, ax2, ax3, ax4) = axes

    # Title & Speed
    ax1.plot(df['Relative_Time'], df['Speed_m_s'], color='purple', alpha=0.4)
    ax1.set_title(f"Friction Analysis: {os.path.basename(latest_file)} | mu_d = {mu_d:.3f}", fontsize=14, fontweight='bold')
    
    # --- ADD WATERMARK IF FAIL ---
    if mu_d >= MU_STATIC_LIMIT:
        fig.text(0.5, 0.5, 'DATA INVALID / FAIL\nEXCEEDS STATIC FRICTION', 
                 fontsize=50, color='red', alpha=0.3,
                 ha='center', va='center', rotation=30, fontweight='bold')
    
    # Position Plots
    ax2.plot(df['Relative_Time'], df['Px'], color='crimson')
    ax2.set_ylabel('Red (X) - Slide (m)')
    ax3.plot(df['Relative_Time'], df['Py'], color='forestgreen')
    ax3.set_ylabel('Green (Y) - Height (m)')
    ax4.plot(df['Relative_Time'], df['Pz'], color='dodgerblue')
    ax4.set_ylabel('Blue (Z) - Drift (m)')
    ax4.set_xlabel('Time (seconds)')

    # Highlight the "Jump"
    t1, t2 = df.loc[start_idx, 'Relative_Time'], df.loc[end_idx, 'Relative_Time']
    for ax in axes:
        ax.axvspan(t1, t2, color='orange', alpha=0.15)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"🖼️  Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()