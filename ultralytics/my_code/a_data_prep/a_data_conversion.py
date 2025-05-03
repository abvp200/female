# Taking dataset in COCO format - prepared for pytorch-models-evaluation
# and converting to ultralitics format.

import os
import json
import shutil
from tqdm import tqdm


for split_no in [0,1,2,3,4]:
    print('='*80)
    print(f'SPLIT #{split_no}')
    print('-'*80)

    # Define the paths
    root_src = f'/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/yolo_v6_all_classes_COCO_v4_noEmpty/repeated_split_{split_no}/'
    root_target = f'/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/ultralitics_all_classes_COCO_v4_noEmpty/repeated_split_{split_no}/'

    source_annotations = os.path.join(root_src, 'annotations')
    target_annotations = os.path.join(root_target, 'annotations')

    # Create new directory structure
    os.makedirs(os.path.join(root_target, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(root_target, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(root_target, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(root_target, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(root_target, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(root_target, 'labels', 'test'), exist_ok=True)

    # Function to convert bbox from COCO to YOLO format
    def convert_bbox(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[2] / 2.0) * dw
        y = (box[1] + box[3] / 2.0) * dh
        w = box[2] * dw
        h = box[3] * dh
        return (x, y, w, h)

    # Function to process each json file
    def process_json(json_path, img_dest, label_dest):
        with open(json_path) as f:
            data = json.load(f)
            
        images = {img['id']: img for img in data['images']}
        annotations = data['annotations']
        
        for annotation in tqdm(annotations, desc=f'Processing {os.path.basename(json_path)}'):
            img_id = annotation['image_id']
            img_info = images[img_id]
            
            img_filename = os.path.basename(img_info['file_name'])
            img_src = os.path.join(root_src, 'images', img_filename)
            img_dst = os.path.join(img_dest, img_filename)
            
            # Copy image symlink
            if not os.path.exists(img_dst):  # Avoid copying multiple times
                if os.path.islink(img_src):
                    target_path = os.readlink(img_src)
                    os.symlink(target_path, img_dst)
                else:
                    shutil.copy2(img_src, img_dst)
            
            # Create label file
            label_filename = img_filename.replace('.jpg', '.txt')
            label_path = os.path.join(label_dest, label_filename)
            
            with open(label_path, 'a') as label_file:
                yolo_bbox = convert_bbox((img_info['width'], img_info['height']), annotation['bbox'])
                label_file.write(f"{annotation['category_id']} {' '.join(map(str, yolo_bbox))}\n")

    # Process each dataset split
    process_json(os.path.join(root_src, 'annotations', 'instances_train.json'), 
                os.path.join(root_target, 'images', 'train'), 
                os.path.join(root_target, 'labels', 'train'))

    process_json(os.path.join(root_src, 'annotations', 'instances_val.json'), 
                os.path.join(root_target, 'images', 'val'), 
                os.path.join(root_target, 'labels', 'val'))

    process_json(os.path.join(root_src, 'annotations', 'instances_test.json'), 
                os.path.join(root_target, 'images', 'test'), 
                os.path.join(root_target, 'labels', 'test'))

    # Copy annotations folder
    shutil.copytree(source_annotations, target_annotations)

    # Generate the data.yml file
    categories = [
        {"id": 0, "name": "Adhesions.Dense"},
        {"id": 1, "name": "Adhesions.Filmy"},
        {"id": 2, "name": "Deep Endometriosis"},
        {"id": 3, "name": "Ovarian.Chocolate Fluid"},
        {"id": 4, "name": "Ovarian.Endometrioma"},
        {"id": 5, "name": "Ovarian.Endometrioma[B]"},
        {"id": 6, "name": "Superficial.Black"},
        {"id": 7, "name": "Superficial.Red"},
        {"id": 8, "name": "Superficial.Subtle"},
        {"id": 9, "name": "Superficial.White"},
    ]

    data_yml_content = f"""
# Train/val/test sets
path: /mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/ultralitics_all_classes_COCO_v4_noEmpty/repeated_split_{split_no} # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')
test: images/test # test images (optional)

# Classes
names:
"""

    for category in categories:
        data_yml_content += f"  {category['id']}: {category['name']}\n"

    with open(os.path.join(root_target, 'data.yml'), 'w') as f:
        f.write(data_yml_content)


    # Now create annotations folder in root_target
