import os
import shutil
from datetime import datetime

# OBJECT="nolid_cube"
# OBJECT="blue_cylinder"
OBJECT="colored_cubes"
# OBJECT="wooden_cube"

# SURFACE="green_rub"
SURFACE="wood_smooth"
# SURFACE="wood_rough"

def archive():
    # Create a unique folder for this entire experiment batch
    batch_name = f"exp_{OBJECT}_{SURFACE}"
    target_dir = os.path.join("results", batch_name)
    
    # Folders to move
    # folders_to_move = ['aruco_velocity', 'videos', 'plots']
    folders_to_move = ['aruco_velocity', 'videos']
    # Files to copy
    # files_to_copy = ['experiment_summary_final.csv']
    files_to_copy = ['"incline_experiment_summary.csv"']
    
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"📦 Archiving current batch to: {target_dir}")
    
    for folder in folders_to_move:
        if os.path.exists(folder):
            shutil.move(folder, os.path.join(target_dir, folder))
            print(f"  - Moved {folder}/")
            
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(target_dir, file))
            print(f"  - Copied {file}")

    print("\n✅ Done! Your main workspace is now clean and ready for new trials.")

if __name__ == "__main__":
    archive()