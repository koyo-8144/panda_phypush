import pandas as pd
import numpy as np
import glob
import os

G = 9.81 
MU_STATIC_LIMIT = 0.4

def calculate_single_friction(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 5: return None
        
        df['jump_x'] = df['Px'].diff().abs()
        jump_idx = df['jump_x'].idxmax()
        
        if df.loc[jump_idx, 'jump_x'] < 0.05: return None

        start_idx = max(0, jump_idx - 1)
        end_idx = jump_idx
        
        dt = df.loc[end_idx, 'Timestamp'] - df.loc[start_idx, 'Timestamp']
        p1 = df.loc[start_idx, ['Px', 'Pz']].values
        p2 = df.loc[end_idx, ['Px', 'Pz']].values
        
        d = np.linalg.norm(p2 - p1)
        v_0 = (2 * d) / dt if dt > 0 else 0
        mu_d = (v_0**2) / (2 * G * d) if d > 0 else 0
        
        is_valid = mu_d < MU_STATIC_LIMIT
        
        return {
            "Filename": os.path.basename(file_path),
            "Distance_m": round(d, 4),
            "mu_d": round(mu_d, 4),
            "Valid": is_valid
        }
    except Exception:
        return None

def main():
    csv_folder = "aruco_velocity"
    video_folder = "videos"
    list_of_files = glob.glob(f"{csv_folder}/*.csv")
    
    if not list_of_files: return

    all_results = [calculate_single_friction(f) for f in list_of_files if calculate_single_friction(f)]
    if not all_results: return

    df_all = pd.DataFrame(all_results)
    valid_trials = df_all[df_all['Valid'] == True]
    invalid_trials = df_all[df_all['Valid'] == False]

    # --- 1. Statistics Calculation ---
    mean_mu = valid_trials['mu_d'].mean() if not valid_trials.empty else 0
    std_mu = valid_trials['mu_d'].std() if not valid_trials.empty else 0
    rel_error = (std_mu / mean_mu) * 100 if mean_mu != 0 else 0

    # --- 2. Print Summary to Terminal ---
    print("\n" + "="*60)
    print("🔬 FILTERED EXPERIMENT SUMMARY")
    print("="*60)
    print(df_all.to_string(index=False))
    print("-" * 60)
    print(f"Average μd: {mean_mu:.4f} | StdDev: {std_mu:.4f} | Error: {rel_error:.2f}%")
    print("="*60)

    # --- 3. Save Detailed CSV with Stats at the Bottom ---
    summary_path = "experiment_summary_final.csv"
    df_all.to_csv(summary_path, index=False)
    
    with open(summary_path, 'a') as f:
        f.write("\n--- STATISTICS (VALID TRIALS ONLY) ---\n")
        f.write(f"Valid Trials,{len(valid_trials)} of {len(df_all)}\n")
        f.write(f"Clean Average mu_d,{mean_mu:.4f}\n")
        f.write(f"Clean Std Deviation,{std_mu:.4f}\n")
        f.write(f"New Relative Error,{rel_error:.2f}%\n")

    print(f"📁 Detailed report saved to: {summary_path}")

    # --- 4. Cleanup Files for Invalid Trials ---
    if not invalid_trials.empty:
        print(f"\n🧹 Cleaning up {len(invalid_trials)} invalid files...")
        for _, row in invalid_trials.iterrows():
            csv_file = os.path.join(csv_folder, row['Filename'])
            vid_file = os.path.join(video_folder, row['Filename'].replace('velocity_', 'tracking_').replace('.csv', '.mp4'))
            
            if os.path.exists(csv_file): os.remove(csv_file)
            if os.path.exists(vid_file): os.remove(vid_file)
        print("✅ Workspace cleaned.")

if __name__ == "__main__":
    main()