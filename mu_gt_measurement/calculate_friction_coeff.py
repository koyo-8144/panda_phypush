import pandas as pd
import numpy as np
import glob
import os

G = 9.81 
MU_STATIC_LIMIT = 0.4  # Your measured static friction limit

def main():
    csv_folder = "aruco_velocity"
    list_of_files = glob.glob(f"{csv_folder}/*.csv")
    if not list_of_files: return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"📊 Analyzing: {latest_file}")
    df = pd.read_csv(latest_file)
    
    df['jump_x'] = df['Px'].diff().abs()
    jump_idx = df['jump_x'].idxmax()
    
    if df.loc[jump_idx, 'jump_x'] < 0.05:
        print("❌ No significant slide detected.")
        return

    start_idx = max(0, jump_idx - 1)
    end_idx = jump_idx
    
    dt = df.loc[end_idx, 'Timestamp'] - df.loc[start_idx, 'Timestamp']
    p1 = df.loc[start_idx, ['Px', 'Pz']].values
    p2 = df.loc[end_idx, ['Px', 'Pz']].values
    
    d = np.linalg.norm(p2 - p1)
    v_0 = (2 * d) / dt if dt > 0 else 0
    mu_d = (v_0**2) / (2 * G * d) if d > 0 else 0

    # Status Check
    status = "✅ PASS" if mu_d < MU_STATIC_LIMIT else "❌ FAIL (Exceeds Static Friction)"

    print(f"\n--- ⚙️ ENDPOINT PHYSICS RESULTS ⚙️ ---")
    print(f"Status:              {status}")
    print(f"Slide Distance (d):  {d:.4f} m")
    print(f"Estimated v_0:       {v_0:.4f} m/s")
    print(f"------------------------------------")
    print(f"Calculated mu_d:     {mu_d:.4f}")
    print(f"------------------------------------")

if __name__ == "__main__":
    main()