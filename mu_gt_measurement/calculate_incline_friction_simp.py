import pandas as pd
import numpy as np
import math

# --- EXPERIMENT CONFIGURATION ---
G = 9.81
THETA_DEGREES = 33.0  # Your measured ramp angle
THETA_SPECIAL = 31.5
THETA_SPECIAL_v2 = 30.6

NO_LID_CUBE_LEN = 0.15
BLUE_CYLINDER_LEN = 0.1
COLORED_CUBES_LEN = 0.13
WOODEN_CUBE_LEN = 0.139
WOODEN_CUBE_EVEN_LEN = 0.141

DISTANCE_M_GREEN_RUB = 0.90 # top to bottom (0.9 meters)
DISTANCE_M_WOOD_SMOOTH_ROUGH = 1.22  # top to bottom (1.22 meters)

# ⏱️ ENTER YOUR MEASURED TIMES HERE:
# Just type the times (in seconds) you read from your video timestamps for each slide.

# ---------- GREEN RUB ----------
MANUAL_TIMES_NO_LID_CUBE_GREEN_RUB = [
59.87-58.86,
7.95-6.77,
18.34-17.30,
25.27-24.22,
32.21-31.21,
39.19-38.21,
45.90-44.93,
52.87-51.87,
60.32-59.37,
6.83-5.94,
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

MANUAL_TIMES_WOODEN_CUBE_EVEN_GREEN_RUB = [
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
28.94-28.15,
34.99-34.24,
41.17-40.39,
47.45-46.65,
53.84-53.14,
60.07-59.38,
6.32-5.55,
12.77-12.00,
19.48-18.75,
26.56-25.83,
]

MANUAL_TIMES_WOODEN_CUBE_EVEN_WOOD_SMOOTH = [
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
10.24-9.42,
16.25-15.42,
22.52-21.72,
28.64-27.81,
35.01-34.24,
40.65-39.82,
47.60-46.73,
53.58-52.74,
60.05-59.26,
6.63-5.87,
]

MANUAL_TIMES_BLUE_CYLINDER_WOOD_ROUGH = [
36.89-36.08,
42.60-41.73,
48.66-47.82,
54.77-53.76,
60.65-59.59,
6.69-5.77,
12.87-11.88,
19.31-18.16,
25.71-24.31,
31.73-30.30,
]

MANUAL_TIMES_COLORED_CUBES_WOOD_ROUGH = [
19.05-18.30,
28.05-27.29,
34.16-33.42,
42.30-41.53,
48.34-47.59,
54.91-54.19,
0.85-0.09,
7.63-6.91,
13.46-12.69,
57.33-56.55,
]

MANUAL_TIMES_WOODEN_CUBE_WOOD_ROUGH = [
46.11-45.32,
52.92-52.15,
59.47-58.69,
5.87-5.16,
12.42-11.68,
18.69-17.94,
25.23-24.52,
31.30-30.55,
37.54-36.86,
43.70-43.01,
]

MANUAL_TIMES_WOODEN_CUBE_EVEN_WOOD_ROUGH = [
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
    # 1. Map all data continuously without toggle checks
    dist_gr = DISTANCE_M_GREEN_RUB
    dist_wsr = DISTANCE_M_WOOD_SMOOTH_ROUGH
    
    experiments = [
        # GREEN RUB (Notice the special angle 32.0 assigned here!)
        {"surface": "Green Rub", "object": "No Lid Cube", "angle": THETA_SPECIAL, "dist": dist_gr - NO_LID_CUBE_LEN, "times": MANUAL_TIMES_NO_LID_CUBE_GREEN_RUB},
        {"surface": "Green Rub", "object": "Blue Cylinder", "angle": THETA_DEGREES, "dist": dist_gr - BLUE_CYLINDER_LEN, "times": MANUAL_TIMES_BLUE_CYLINDER_GREEN_RUB},
        {"surface": "Green Rub", "object": "Colored Cubes", "angle": THETA_DEGREES, "dist": dist_gr - COLORED_CUBES_LEN, "times": MANUAL_TIMES_COLORED_CUBES_GREEN_RUB},
        {"surface": "Green Rub", "object": "Wooden Cube", "angle": THETA_DEGREES, "dist": dist_gr - WOODEN_CUBE_LEN, "times": MANUAL_TIMES_WOODEN_CUBE_GREEN_RUB},
        
        # WOOD SMOOTH
        {"surface": "Wood Smooth", "object": "No Lid Cube", "angle": THETA_DEGREES, "dist": dist_wsr - NO_LID_CUBE_LEN, "times": MANUAL_TIMES_NO_LID_CUBE_WOOD_SMOOTH},
        {"surface": "Wood Smooth", "object": "Blue Cylinder", "angle": THETA_DEGREES, "dist": dist_wsr - BLUE_CYLINDER_LEN, "times": MANUAL_TIMES_BLUE_CYLINDER_WOOD_SMOOTH},
        {"surface": "Wood Smooth", "object": "Colored Cubes", "angle": THETA_DEGREES, "dist": dist_wsr - COLORED_CUBES_LEN, "times": MANUAL_TIMES_COLORED_CUBES_WOOD_SMOOTH},
        {"surface": "Wood Smooth", "object": "Wooden Cube", "angle": THETA_DEGREES, "dist": dist_wsr - WOODEN_CUBE_LEN, "times": MANUAL_TIMES_WOODEN_CUBE_WOOD_SMOOTH},
        
        # WOOD ROUGH
        {"surface": "Wood Rough", "object": "No Lid Cube", "angle": THETA_DEGREES, "dist": dist_wsr - NO_LID_CUBE_LEN, "times": MANUAL_TIMES_NO_LID_CUBE_WOOD_ROUGH},
        {"surface": "Wood Rough", "object": "Blue Cylinder", "angle": THETA_DEGREES, "dist": dist_wsr - BLUE_CYLINDER_LEN, "times": MANUAL_TIMES_BLUE_CYLINDER_WOOD_ROUGH},
        {"surface": "Wood Rough", "object": "Colored Cubes", "angle": THETA_DEGREES, "dist": dist_wsr - COLORED_CUBES_LEN, "times": MANUAL_TIMES_COLORED_CUBES_WOOD_ROUGH},
        {"surface": "Wood Rough", "object": "Wooden Cube", "angle": THETA_DEGREES, "dist": dist_wsr - WOODEN_CUBE_LEN, "times": MANUAL_TIMES_WOODEN_CUBE_WOOD_ROUGH},
    ]

    all_results = []
    summary_stats = []

    # 2. Process everything
    for exp in experiments:
        if not exp["times"]: 
            continue # Skip empty lists
            
        mu_k_list = []
        
        # Calculate the angle specific to THIS experiment
        theta_rad = math.radians(exp["angle"])
        
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
                "Angle": exp["angle"],
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
                "Angle": exp["angle"],
                "Trials": len(mu_k_list),
                "Avg_mu_k": round(mean_mu, 4),
                "Std_Dev": round(std_mu, 4),
                "Rel_Error_%": round(rel_error, 2)
            })

    # 3. Print & Save
    if not all_results:
        print("❌ No data found to process. Please add times to your lists.")
        return

    df_all = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summary_stats)

    print("\n" + "=" * 80)
    print("🔬 MANUAL INCLINED PLANE SUMMARY")
    print("=" * 80)
    print("\n--- OVERALL STATISTICS ---")
    print(df_summary.to_string(index=False))
    
    print("\n--- INDIVIDUAL TRIALS ---")
    # Uncomment the line below if you want the massive list of individual trials printed to the terminal too
    # print(df_all.to_string(index=False)) 
    print("=" * 80)

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