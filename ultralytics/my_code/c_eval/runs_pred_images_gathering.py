import os
import shutil
from pathlib import Path

# Source directory containing all the train folders
source_dir = "/mnt/runs/detect"

# Destination directory where all files will be gathered
dest_dir = "/mnt/runs/gathered_val_pred_images"

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Iterate through all subdirectories in source_dir
for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)
    
    # Check if it's a directory and starts with 'train'
    if os.path.isdir(subdir_path) and subdir.startswith('train'):
        # Iterate through files in the subdirectory
        for filename in os.listdir(subdir_path):
            # Check if file matches our pattern
            if filename.startswith('val_batch') and (filename.endswith('_labels.jpg') or filename.endswith('_pred.jpg')):
                # Create new filename with subfolder prefix
                new_filename = f"{subdir}_{filename}"
                
                # Full paths for source and destination
                src_file = os.path.join(subdir_path, filename)
                dest_file = os.path.join(dest_dir, new_filename)
                
                # Copy the file
                shutil.copy2(src_file, dest_file)
                print(f"Copied: {src_file} -> {dest_file}")

print("All files have been gathered and renamed.")