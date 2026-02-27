import pandas as pd
import numpy as np
import math

# --- EXPERIMENT CONFIGURATION ---
G = 9.81
THETA_DEGREES = 33.0  # Your measured ramp angle

GREEN_RUB = 1
WOOD_SMOOTH = 0
WOOD_ROUGH = 0

NO_LID_CUBE_LEN = 0.15
BLUE_CYLINDER_LEN = 0.15
COLORED_CUBES_LEN = 0.15
WOODEN_CUBE_LEN = 0.15

DISTANCE_M_GREEN_RUB = 0.90 # top to bottom (0.9 meters)
DISTANCE_M_WOOD_SMOOTH_ROUGH = 1.22  # top to bottom (1.22 meters)

# ⏱️ ENTER YOUR MEASURED TIMES HERE:
# Just type the times (in seconds) you read from your video timestamps for each slide.

# ---------- GREEN RUB ----------
MANUAL_TIMES_NO_LID_CUBE_GREEN_RUB = [
25.30-24.12,
33.14-32.05,
41.68-40.50,
53.07-51.94,
20.77-19.57,
18.57-17.47,
25.85-24.68,
37.05-35.90,
50.43-49.28,
11.47-10.21,
]

MANUAL_TIMES_BLUE_CYLINDER_GREEN_RUB = [
47.42-46.63,
2.46-1.64,
14.77-13.92,
29.58-28.82,
44.44-43.56,
58.10-57.26,
11.12-10.31,
23.55-22.71,
35.81-34.99,
49.58-48.69,
]

MANUAL_TIMES_COLORED_CUBES_GREEN_RUB = [
51.02-50.15,
2.79-2.05,
14.57-13.83,
28.61-27.87,
39.72-39.03,
53.29-52.56,
6.68-5.98,
19.98-19.30,
33.86-33.14,  
48.60-47.86,  
]

MANUAL_TIMES_WOODEN_CUBE_GREEN_RUB = [
46.31-45.59,
53.61-52.87,
0.93-0.16,
11.42-10.72,
18.66-17.95,
25.98-25.24,
32.91-32.12,
40.72-39.98,
49.48-48.72,
0.88-0.15,
]

# ---------- WOOD SMOOTH ----------
MANUAL_TIMES_NO_LID_CUBE_WOOD_SMOOTH = [
27.80-26.86,
34.23-33.26,
41.16-40.32,
47.85-46.90,
54.59-53.74,
1.21-0.26,
7.75-6.87,
14.51-13.55,
21.87-21.00,
29.85-28.80,
]

MANUAL_TIMES_BLUE_CYLINDER_WOOD_SMOOTH = [
36.70-35.87,
44.62-43.83,
51.87-51.05,
58.58-57.72,
5.81-5.01,
12.14-11.27,
17.96-17.10,
24.49-23.63,
31.17-30.27,
17.08-16.14,
]

MANUAL_TIMES_COLORED_CUBES_WOOD_SMOOTH = [
17.39-16.63,
25.38-24.67,
31.43-30.69,
36.90-36.16,
42.92-42.21,
49.77-49.05,
55.66-54.90,
2.16-1.44,
9.67-8.93,
17.59-16.84,
]

MANUAL_TIMES_WOODEN_CUBE_WOOD_SMOOTH = [
-,
-,
-,
-,
-,
-,
-,
-,
-,
-,
]

# ---------- WOOD ROUGH ----------
MANUAL_TIMES_NO_LID_CUBE_WOOD_ROUGH = [
-,
-,
-,
-,
-,
-,
-,
-,
-,
-,
]

MANUAL_TIMES_BLUE_CYLINDER_WOOD_ROUGH = [
-,
-,
-,
-,
-,
-,
-,
-,
-,
-,
]

MANUAL_TIMES_COLORED_CUBES_WOOD_ROUGH = [
-,
-,
-,
-,
-,
-,
-,
-,
-,
-,
]

MANUAL_TIMES_WOODEN_CUBE_WOOD_ROUGH = [
-,
-,
-,
-,
-,
-,
-,
-,
-,
-,
]


def main():
    # 1. Map all your data into a structure we can loop through
    experiments = []
    
    if GREEN_RUB:
        dist = DISTANCE_M_GREEN_RUB
        experiments.extend([
            {"surface": "Green Rub", "object": "No Lid Cube", "dist": dist - NO_LID_CUBE_LEN, "times": MANUAL_TIMES_NO_LID_CUBE_GREEN_RUB},
            {"surface": "Green Rub", "object": "Blue Cylinder", "dist": dist - BLUE_CYLINDER_LEN, "times": MANUAL_TIMES_BLUE_CYLINDER_GREEN_RUB},
            {"surface": "Green Rub", "object": "Colored Cubes", "dist": dist - COLORED_CUBES_LEN, "times": MANUAL_TIMES_COLORED_CUBES_GREEN_RUB},
            {"surface": "Green Rub", "object": "Wooden Cube", "dist": dist - WOODEN_CUBE_LEN, "times": MANUAL_TIMES_WOODEN_CUBE_GREEN_RUB},
        ])
        
    if WOOD_SMOOTH:
        dist = DISTANCE_M_WOOD_SMOOTH_ROUGH
        experiments.extend([
            {"surface": "Wood Smooth", "object": "No Lid Cube", "dist": dist - NO_LID_CUBE_LEN, "times": MANUAL_TIMES_NO_LID_CUBE_WOOD_SMOOTH},
            {"surface": "Wood Smooth", "object": "Blue Cylinder", "dist": dist - BLUE_CYLINDER_LEN, "times": MANUAL_TIMES_BLUE_CYLINDER_WOOD_SMOOTH},
            {"surface": "Wood Smooth", "object": "Colored Cubes", "dist": dist - COLORED_CUBES_LEN, "times": MANUAL_TIMES_COLORED_CUBES_WOOD_SMOOTH},
            {"surface": "Wood Smooth", "object": "Wooden Cube", "dist": dist - WOODEN_CUBE_LEN, "times": MANUAL_TIMES_WOODEN_CUBE_WOOD_SMOOTH},
        ])
        
    if WOOD_ROUGH:
        dist = DISTANCE_M_WOOD_SMOOTH_ROUGH
        experiments.extend([
            {"surface": "Wood Rough", "object": "No Lid Cube", "dist": dist - NO_LID_CUBE_LEN, "times": MANUAL_TIMES_NO_LID_CUBE_WOOD_ROUGH},
            {"surface": "Wood Rough", "object": "Blue Cylinder", "dist": dist - BLUE_CYLINDER_LEN, "times": MANUAL_TIMES_BLUE_CYLINDER_WOOD_ROUGH},
            {"surface": "Wood Rough", "object": "Colored Cubes", "dist": dist - COLORED_CUBES_LEN, "times": MANUAL_TIMES_COLORED_CUBES_WOOD_ROUGH},
            {"surface": "Wood Rough", "object": "Wooden Cube", "dist": dist - WOODEN_CUBE_LEN, "times": MANUAL_TIMES_WOODEN_CUBE_WOOD_ROUGH},
        ])

    theta_rad = math.radians(THETA_DEGREES)
    all_results = []
    summary_stats = []

    # 2. Process everything
    for exp in experiments:
        if not exp["times"]: 
            continue # Skip empty lists
            
        mu_k_list = []
        
        for i, t in enumerate(exp["times"]):
            if t <= 0: continue

            # Kinematics & Forces Math
            a = (2 * exp["dist"]) / (t**2)
            mu_k = np.tan(theta_rad) - (a / (G * np.cos(theta_rad)))
            mu_k_list.append(mu_k)

            all_results.append({
                "Surface": exp["surface"],
                "Object": exp["object"],
                "Trial": i + 1,
                "Distance_m": round(exp["dist"], 4),
                "Time_s": round(t, 4),
                "Accel_m_s2": round(a, 4),
                "mu_k": round(mu_k, 4),
            })

        # Calculate Group Statistics
        if mu_k_list:
            mean_mu = np.mean(mu_k_list)
            std_mu = np.std(mu_k_list, ddof=1) if len(mu_k_list) > 1 else 0
            rel_error = (std_mu / mean_mu) * 100 if mean_mu != 0 else 0
            
            summary_stats.append({
                "Surface": exp["surface"],
                "Object": exp["object"],
                "Trials": len(mu_k_list),
                "Avg_mu_k": round(mean_mu, 4),
                "Std_Dev": round(std_mu, 4),
                "Rel_Error_%": round(rel_error, 2)
            })

    # 3. Print & Save
    if not all_results:
        print("❌ No data found to process. Please add times to your active lists.")
        return

    df_all = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summary_stats)

    print("\n" + "=" * 70)
    print(f"🔬 MANUAL INCLINED PLANE SUMMARY (Angle: {THETA_DEGREES}°)")
    print("=" * 70)
    print("\n--- OVERALL STATISTICS ---")
    print(df_summary.to_string(index=False))
    
    print("\n--- INDIVIDUAL TRIALS ---")
    print(df_all.to_string(index=False))
    print("=" * 70)

    # Save to CSV
    summary_path = "manual_incline_summary.csv"
    df_all.to_csv(summary_path, index=False)
    
    # Append the stats to the bottom of the CSV
    with open(summary_path, 'a') as f:
        f.write("\n\n--- STATISTICS ---\n")
        df_summary.to_csv(f, index=False)
        
    print(f"\n📁 Detailed report saved to: {summary_path}")


if __name__ == "__main__":
    main()