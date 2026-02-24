import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def main():
    # 1. Setup Input and Output Directories
    csv_folder = "aruco_velocity"
    plots_folder = "plots"
    os.makedirs(plots_folder, exist_ok=True)
    
    list_of_files = glob.glob(f"{csv_folder}/*.csv")
    
    if not list_of_files:
        print(f"❌ No CSV files found in the '{csv_folder}' directory.")
        return
        
    # Find the newest CSV file
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"📊 Loading data from: {latest_file}")

    # Generate the output filename for the plot (matches the CSV filename)
    base_filename = os.path.basename(latest_file).replace('.csv', '.png')
    plot_filepath = os.path.join(plots_folder, base_filename)

    # 2. Read the data
    df = pd.read_csv(latest_file)

    # Calculate relative time (starts at T=0)
    df['Relative_Time'] = df['Timestamp'] - df['Timestamp'].iloc[0]

    # 3. Find Start and Stop Points
    # Find the release point (v_0)
    smoothed_speed = df['Speed_m_s'].rolling(window=3, center=True).mean().fillna(df['Speed_m_s'])
    start_idx = smoothed_speed.idxmax()
    start_time = df.loc[start_idx, 'Relative_Time']

    # Find the stop point (drops below 0.02 m/s)
    stop_threshold = 0.02
    post_release_data = df.loc[start_idx:]
    stop_indices = post_release_data[post_release_data['Speed_m_s'] < stop_threshold].index
    
    if len(stop_indices) == 0:
        stop_idx = df.index[-1] 
    else:
        stop_idx = stop_indices[0]
    stop_time = df.loc[stop_idx, 'Relative_Time']

    # 4. Create a 4x1 stacked Dashboard sharing the X (time) axis
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    (ax1, ax2, ax3, ax4) = axes

    # --- Plot 1: Speed vs Time ---
    ax1.plot(df['Relative_Time'], df['Speed_m_s'], color='purple', linewidth=2)
    ax1.plot(start_time, df.loc[start_idx, 'Speed_m_s'], 'ro', markersize=8, label='Release Point (v_0)')
    ax1.plot(stop_time, df.loc[stop_idx, 'Speed_m_s'], 'bs', markersize=8, label='Stop Point')
    ax1.set_title('Speed & Position Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Speed (m/s)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')

    # --- Plot 2: X Position (Up/Down) vs Time ---
    ax2.plot(df['Relative_Time'], df['Px'], color='crimson', linewidth=2)
    ax2.plot(start_time, df.loc[start_idx, 'Px'], 'ro', markersize=8)
    ax2.plot(stop_time, df.loc[stop_idx, 'Px'], 'bs', markersize=8)
    ax2.set_ylabel('Red (X) - Up (m)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # --- Plot 3: Y Position (Slide) vs Time ---
    ax3.plot(df['Relative_Time'], df['Py'], color='forestgreen', linewidth=2)
    ax3.plot(start_time, df.loc[start_idx, 'Py'], 'ro', markersize=8)
    ax3.plot(stop_time, df.loc[stop_idx, 'Py'], 'bs', markersize=8)
    ax3.set_ylabel('Green (Y) - Slide (m)', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # --- Plot 4: Z Position (Drift) vs Time ---
    ax4.plot(df['Relative_Time'], df['Pz'], color='dodgerblue', linewidth=2)
    ax4.plot(start_time, df.loc[start_idx, 'Pz'], 'ro', markersize=8)
    ax4.plot(stop_time, df.loc[stop_idx, 'Pz'], 'bs', markersize=8)
    ax4.set_xlabel('Time (seconds)', fontsize=12)
    ax4.set_ylabel('Blue (Z) - Drift (m)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.7)

    # Draw vertical bounds for the sliding phase across all plots
    for ax in axes:
        ax.axvline(x=start_time, color='orange', linestyle='--', alpha=0.7)
        ax.axvline(x=stop_time, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # 5. Save the plot to the plots directory BEFORE showing it
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    print(f"🖼️  Plot successfully saved to: {plot_filepath}")

    # Display the plots
    plt.show()

if __name__ == "__main__":
    main()