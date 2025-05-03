"""
Some time ago we have extracted images/frames from videos, but used CV2 
and now all images have blue-tint, need to convert them back to RGB.

This script does that.
"""

import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool, Manager

def convert_image(args):
    image_file, input_folder, output_folder, progress_queue = args
    # Read image in BGR format
    bgr_image = cv2.imread(os.path.join(input_folder, image_file))
    
    if bgr_image is not None:
        # Convert BGR image to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # Write RGB image to output folder
        output_file = os.path.join(output_folder, image_file)
        cv2.imwrite(output_file, rgb_image)
    else:
        print(f"Failed to read image file: {image_file}")
    
    progress_queue.put(1)

def convert_bgr_to_rgb(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get list of image files in input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Use multiprocessing Pool to parallelize conversion
    progress_queue = Manager().Queue()
    with Pool() as pool, tqdm(total=len(image_files), desc='converting bgr images to rgb') as pbar:
        for _ in pool.imap_unordered(convert_image, [(image_file, input_folder, output_folder, progress_queue) for image_file in image_files]):
            pbar.update(progress_queue.get())


if __name__ == "__main__":
    input_folder = "/mnt/data_vilen/01_raw/BoundingBox_2022_10_FullUpdated/img"  # Folder containing images in BGR format
    output_folder = "/mnt/data_vilen/01_raw/BoundingBox_2022_10_FullUpdated/img2"  # Folder to save images in RGB format
    
    convert_bgr_to_rgb(input_folder, output_folder)
