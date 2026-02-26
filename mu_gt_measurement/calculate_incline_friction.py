import pandas as pd
import numpy as np
import glob
import os
import math

# --- EXPERIMENT CONFIGURATION ---
G = 9.81 
THETA_DEGREES = 15.0

def calculate_incline_friction(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 10: return None
        
        # 1. Calculate the total 3D displacement from the starting position
        # We use full 3D (X, Y, Z) because the ramp is tilted.
        start_x, start_y, start_z = df['Px'].iloc[0], df['Py'].iloc[0], df['Pz'].iloc[0]
        df['Displacement'] = np.sqrt((df['Px'] - start_x)**2 + 
                                     (df['Py'] - start_y)**2 + 
                                     (df['Pz'] - start_z)**2)
        
        # 2. Robust Start/Stop Detection
        # Start: The last frame before the object moved more than 1 cm (0.01m)
        pre_movement = df[df['Displacement'] < 0.01]
        if pre_movement.empty: return None
        start_idx = pre_movement.index[-1]
        
        # End: The frame where it reached its maximum displacement (bottom of the ramp)
        end_idx = df['Displacement'].idxmax()
        
        if end_idx <= start_idx: return None

        # 3. Extract time (t) and distance (d)
        t = df.loc[end_idx, 'Timestamp'] - df.loc[start_idx, 'Timestamp']
        
        p1 = df.loc[start_idx, ['Px', 'Py', 'Pz']].values
        p2 = df.loc[end_idx, ['Px', 'Py', 'Pz']].values
        d = np.linalg.norm(p2 - p1)
        
        # Filter out tiny accidental bumps
        if d < 0.10 or t <= 0: return None

        # 4. Math: Kinematics & Forces
        # Convert angle to radians for numpy math
        theta_rad = math.radians(THETA_DEGREES)
        
        # a = 2d / t^2
        a = (2 * d) / (t**2)
        
        # mu_k = tan(theta) - a / (g * cos(theta))
        mu_k = np.tan(theta_rad) - (a / (G * np.cos(theta_rad)))
        
        return {
            "Filename": os.path.basename(file_path),
            "Angle_deg": THETA_DEGREES,
            "Distance_m": round(d, 4),
            "Time_s": round(t, 4),
            "Accel_m_s2": round(a, 4),
            "mu_k": round(mu_k, 4)
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    csv_folder = "aruco_velocity"
    list_of_files = glob.glob(f"{csv_folder}/*.csv")
    
    if not list_of_files:
        print("❌ No CSV files found.")
        return

    print(f"📐 Calculating Inclined Plane Friction (Angle: {THETA_DEGREES}°)")
    all_results = [calculate_incline_friction(f) for f in list_of_files]
    all_results = [r for r in all_results if r is not None]

    if not all_results:
        print("❌ No valid slides detected. Ensure the object slid at least 10cm.")
        return

    df_all = pd.DataFrame(all_results)
    
    # Statistics
    mean_mu = df_all['mu_k'].mean()
    std_mu = df_all['mu_k'].std()
    rel_error = (std_mu / mean_mu) * 100 if mean_mu != 0 else 0

    print("\n" + "="*70)
    print("🔬 INCLINED PLANE SUMMARY REPORT")
    print("="*70)
    print(df_all.to_string(index=False))
    print("-" * 70)
    print(f"Average μk: {mean_mu:.4f} | StdDev: {std_mu:.4f} | Error: {rel_error:.2f}%")
    print("="*70)

    # Save to CSV
    summary_path = "incline_experiment_summary.csv"
    df_all.to_csv(summary_path, index=False)
    print(f"\n📁 Report saved to: {summary_path}")

if __name__ == "__main__":
    main()