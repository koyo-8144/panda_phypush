import pandas as pd
import numpy as np
import glob
import os

# Gravity constant (m/s^2)
G = 9.81 

def main():
    # 1. Find the newest CSV file
    csv_folder = "aruco_velocity"
    list_of_files = glob.glob(f"{csv_folder}/*.csv")
    
    if not list_of_files:
        print(f"❌ No CSV files found in the '{csv_folder}' directory.")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"📊 Analyzing data from: {latest_file}\n")

    # 2. Read the data
    df = pd.read_csv(latest_file)
    
    # Smooth the speed slightly to find the true release point (filters out single-frame camera jitter)
    df['Smoothed_Speed'] = df['Speed_m_s'].rolling(window=3, center=True).mean().fillna(df['Speed_m_s'])

    # 3. Find v_0 (The moment of release)
    start_idx = df['Smoothed_Speed'].idxmax()
    v_0 = df.loc[start_idx, 'Smoothed_Speed']
    start_time = df.loc[start_idx, 'Timestamp']
    
    # CHANGED: Extract ONLY the Green (Y) and Blue (Z) coordinates for the starting point
    p1_2d = df.loc[start_idx, ['Py', 'Pz']].values

    # 4. Find the stopping point
    stop_threshold = 0.02 # m/s
    post_release_data = df.loc[start_idx:]
    
    # Find the first frame after release where speed drops below the threshold
    stop_indices = post_release_data[post_release_data['Smoothed_Speed'] < stop_threshold].index
    
    if len(stop_indices) == 0:
        stop_idx = df.index[-1] 
    else:
        stop_idx = stop_indices[0]
        
    stop_time = df.loc[stop_idx, 'Timestamp']
    
    # CHANGED: Extract ONLY the Green (Y) and Blue (Z) coordinates for the stopping point
    p2_2d = df.loc[stop_idx, ['Py', 'Pz']].values

    # 5. Calculate sliding distance (d) using strictly 2D Euclidean distance on the sliding plane
    # Formula: sqrt((Py2 - Py1)^2 + (Pz2 - Pz1)^2)
    d = np.linalg.norm(p2_2d - p1_2d)

    # 6. Calculate Dynamic Friction Coefficient (mu_d)
    if d > 0:
        mu_d = (v_0**2) / (2 * G * d)
    else:
        mu_d = 0.0

    # --- Output the Results ---
    print("--- ⚙️ KINETIC FRICTION RESULTS (2D PLANE) ⚙️ ---")
    print(f"Release Time:      T={start_time:.2f}s")
    print(f"Stop Time:         T={stop_time:.2f}s")
    print(f"Start Pos (P1):    Y:{p1_2d[0]:.3f} Z:{p1_2d[1]:.3f} m")
    print(f"Stop Pos  (P2):    Y:{p2_2d[0]:.3f} Z:{p2_2d[1]:.3f} m")
    print(f"Initial Vel (v_0): {v_0:.4f} m/s")
    print(f"Slide Distance(d): {d:.4f} m (on Y-Z plane)")
    print(f"-------------------------------------------")
    print(f"Calculated mu_d:   {mu_d:.4f}")
    print(f"-------------------------------------------")

if __name__ == "__main__":
    main()