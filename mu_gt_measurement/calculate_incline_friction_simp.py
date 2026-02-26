import pandas as pd
import numpy as np
import math

# --- EXPERIMENT CONFIGURATION ---
G = 9.81
THETA_DEGREES = 33.0  # Your measured ramp angle

GREEN_RUB = 1
WOOD_SMOOTH = 0
WOOD_ROUGH = 0

DISTANCE_M_GREEN_RUB = 0.90 # top to bottom (0.9 meters)
DISTANCE_M_WOOD_SMOOTH_ROUGH = 1.22  # top to bottom (1.22 meters)

# ⏱️ ENTER YOUR MEASURED TIMES HERE:
# Just type the times (in seconds) you read from your video timestamps for each slide.

# ---------- GREEN RUB ----------
MANUAL_TIMES_NO_LID_CUBE_GREEN_RUB = [

]

MANUAL_TIMES_BLUE_CYLINDER_GREEN_RUB = [
    
]

MANUAL_TIMES_COLORED_CUBES_GREEN_RUB = [
    
]

MANUAL_TIMES_WOODEN_CUBE_GREEN_RUB = [
    
]

# ---------- WOOD SMOOTH ----------
MANUAL_TIMES_NO_LID_CUBE_WOOD_SMOOTH = [
 
]

MANUAL_TIMES_BLUE_CYLINDER_WOOD_SMOOTH = [
    
]

MANUAL_TIMES_COLORED_CUBES_WOOD_SMOOTH = [
    
]

MANUAL_TIMES_WOODEN_CUBE_WOOD_SMOOTH = [
    
]

# ---------- WOOD ROUGH ----------
MANUAL_TIMES_NO_LID_CUBE_WOOD_ROUGH = [
 
]

MANUAL_TIMES_BLUE_CYLINDER_WOOD_ROUGH = [
    
]

MANUAL_TIMES_COLORED_CUBES_WOOD_ROUGH = [
    
]

MANUAL_TIMES_WOODEN_CUBE_WOOD_ROUGH = [
    
]


def main():
    # if not MANUAL_TIMES_NO_LID_CUBE_GREEN_RUB:
    #     print("❌ Please enter your measured times into the MANUAL_TIMES list.")
    #     return

    print(
        f"📐 Calculating Friction for {len(MANUAL_TIMES_NO_LID_CUBE_GREEN_RUB)} manual trials..."
    )

    theta_rad = math.radians(THETA_DEGREES)
    results = []

    for i, t in enumerate(MANUAL_TIMES_NO_LID_CUBE_GREEN_RUB):
        if t <= 0:
            continue

        # 1. Acceleration: a = 2d / t^2
        a = (2 * DISTANCE_M) / (t**2)

        # 2. Friction Coeff: mu_k = tan(theta) - a / (g * cos(theta))
        mu_k = np.tan(theta_rad) - (a / (G * np.cos(theta_rad)))

        results.append(
            {
                "Trial": i + 1,
                "Distance_m": DISTANCE_M,
                "Time_s": t,
                "Accel_m_s2": round(a, 4),
                "mu_k": round(mu_k, 4),
            }
        )

    # Convert to DataFrame for nice formatting
    df_all = pd.DataFrame(results)

    # Calculate Statistics
    mean_mu = df_all["mu_k"].mean()
    std_mu = df_all["mu_k"].std()
    rel_error = (std_mu / mean_mu) * 100 if mean_mu != 0 else 0

    # Print Report
    print("\n" + "=" * 55)
    print("🔬 MANUAL INCLINED PLANE SUMMARY REPORT")
    print("=" * 55)
    print(df_all.to_string(index=False))
    print("-" * 55)
    print(f"Fixed Distance: {DISTANCE_M} m | Angle: {THETA_DEGREES}°")
    print(f"Average μk:     {mean_mu:.4f}")
    print(f"StdDev:         {std_mu:.4f}")
    print(f"Error:          {rel_error:.2f}%")
    print("=" * 55)

    # Save to CSV
    summary_path = "manual_incline_summary.csv"
    df_all.to_csv(summary_path, index=False)
    print(f"\n📁 Report saved to: {summary_path}")


if __name__ == "__main__":
    main()
