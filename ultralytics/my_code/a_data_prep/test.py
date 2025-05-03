import os
from tqdm.auto import tqdm

processed_lines = []
    
dirpath = '/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/ultralitics_per_class_0_COCO_v4_noEmpty/repeated_split_0/labels/train'
for fname in tqdm(os.listdir(dirpath)):
    fpath = os.path.join(dirpath, fname)


    # Open the text file
    with open(fpath, 'r') as file:
        lines = file.readlines()

    # Process each line to skip the first number and join the rest
    for line in lines:
        parts = line.split()  # Split the line into parts
        if parts:  # Check if the line is not empty
            processed_line = int(parts[0])
            processed_lines.append(processed_line)

    
print(set(processed_lines))
    