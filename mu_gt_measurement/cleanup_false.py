import pandas as pd
import os

def main():
    summary_file = "experiment_summary_filtered.csv"
    csv_folder = "aruco_velocity"
    video_folder = "videos"

    if not os.path.exists(summary_file):
        print(f"❌ Could not find {summary_file}. Please run the summary script first.")
        return

    # Load the summary data
    df = pd.read_csv(summary_file)

    # Filter for the invalid trials
    invalid_trials = df[df['Valid'] == False]

    if invalid_trials.empty:
        print("✨ No invalid trials found to delete!")
        return

    print(f"🧹 Found {len(invalid_trials)} invalid trials. Starting cleanup...\n")

    for _, row in invalid_trials.iterrows():
        csv_filename = row['Filename']
        video_filename = csv_filename.replace('velocity_', 'tracking_').replace('.csv', '.mp4')

        csv_path = os.path.join(csv_folder, csv_filename)
        video_path = os.path.join(video_folder, video_filename)

        # Delete CSV
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"  - Deleted CSV: {csv_filename}")
        else:
            print(f"  - CSV not found (already gone?): {csv_filename}")

        # Delete Video
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"  - Deleted Video: {video_filename}")
        else:
            print(f"  - Video not found: {video_filename}")

    print("\n✅ Cleanup complete. Only your high-quality trials remain.")

if __name__ == "__main__":
    main()