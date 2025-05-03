# Taking dataset in COCO format - prepared for pytorch-models-evaluation
# and converting to ultralitics format.

import os
import json
import shutil
from tqdm import tqdm


# Generate the data.yml file
categories = [
    {"id": 0, "name": "Adhesions.Dense"},
    {"id": 1, "name": "Adhesions.Filmy"},
    {"id": 2, "name": "Deep Endometriosis"},
    {"id": 3, "name": "Ovarian.Chocolate Fluid"},
    {"id": 4, "name": "Ovarian.Endometrioma"},
    # {"id": 5, "name": "Ovarian.Endometrioma[B]"},
    {"id": 5, "name": "Superficial.Black"},
    {"id": 6, "name": "Superficial.Red"},
    {"id": 7, "name": "Superficial.Subtle"},
    {"id": 8, "name": "Superficial.White"},
]

# That data is already mapped (4+5 classes merged)
categories_groups = [[0],[1],[2],[3],[4],[5],[6],[7],[8]]

for group in categories_groups:
    group_str = str(group).replace(',','_').replace('[','').replace(']','')
    
    per_class_categories = [categories[group[0]]]
    per_class_categories[0]['id'] = 0
    
    for split_no in [0,1,2,3,4]:
        print('='*80)
        print(f'SPLIT #{split_no}, Classes Group: {group}')
        print('-'*80)

        # Define the paths
        root_src = f'/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/yolo_v6_all_classes_COCO_v4_noEmpty/repeated_split_{split_no}/'
        root_target = f'/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/ultralitics_per_class_{group_str}_COCO_v4_noEmpty/repeated_split_{split_no}/'

        source_annotations = os.path.join(root_src, 'annotations')
        target_annotations = os.path.join(root_target, 'annotations')

        # Create new directory structure
        os.makedirs(target_annotations, exist_ok=True)
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
        def process_json(json_path, img_dest, label_dest, current_group):
            with open(json_path) as f:
                data = json.load(f)
                
            images = {img['id']: img for img in data['images']}
            # annotations = data['annotations']
            annotations = [ann for ann in data['annotations'] if ann['category_id'] in current_group]
            
            # Filter out images without annotations
            image_ids_with_annotations = {ann['image_id'] for ann in annotations}
            images_to_keep = {img_id: img for img_id, img in images.items() if img_id in image_ids_with_annotations}
            
            filtered_images = list(images_to_keep.values())

            # Write filtered images to new JSON
            new_data = {
                "images": filtered_images,
                "annotations": annotations,
                "categories": [cat for cat in data['categories'] if cat['id'] in current_group]
            }

            # Set all categories to id=0 ( basically we have only one category )
            for category in new_data['categories']:
                category['id'] = 0

            # Write new JSON
            filtered_json_path = os.path.join(target_annotations, os.path.basename(json_path))
            with open(filtered_json_path, 'w') as f:
                json.dump(new_data, f, indent=4)


            # Process images and labels
            for img_info in tqdm(filtered_images):
                img_filename = os.path.basename(img_info['file_name'])

                # Create label file
                label_filename = img_filename.replace('.jpg', '.txt')
                label_path = os.path.join(label_dest, label_filename)
                
                # Filter annotations for this image
                annotations_for_image = [ann for ann in annotations if ann['image_id'] == img_info['id']]

                # Only proceed if there are annotations for this image
                if annotations_for_image:
                    with open(label_path, 'w') as label_file:
                        for annotation in annotations_for_image:
                            yolo_bbox = convert_bbox((img_info['width'], img_info['height']), annotation['bbox'])
                            label_file.write(f"{0} {' '.join(map(str, yolo_bbox))}\n")  # category_id is always 0 since we remap to one class

                    img_src = os.path.join(root_src, 'images', img_filename)
                    img_dst = os.path.join(img_dest, img_filename)
                    
                    # Copy image
                    if not os.path.exists(img_dst):
                        if os.path.islink(img_src):
                            target_path = os.readlink(img_src)
                            os.symlink(target_path, img_dst)
                        else:
                            shutil.copy2(img_src, img_dst)
                        

        # Process each dataset split
        process_json(os.path.join(root_src, 'annotations', 'instances_train.json'), 
                    os.path.join(root_target, 'images', 'train'), 
                    os.path.join(root_target, 'labels', 'train'),
                    group)

        process_json(os.path.join(root_src, 'annotations', 'instances_val.json'), 
                    os.path.join(root_target, 'images', 'val'), 
                    os.path.join(root_target, 'labels', 'val'),
                    group)

        process_json(os.path.join(root_src, 'annotations', 'instances_test.json'), 
                    os.path.join(root_target, 'images', 'test'), 
                    os.path.join(root_target, 'labels', 'test'),
                    group)

        # # Copy annotations folder
        # shutil.copytree(source_annotations, target_annotations)

        

        data_yml_content = f"""
# Train/val/test sets
path: {root_target} # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')
test: images/test # test images (optional)

# Classes
names:
"""

        for category in per_class_categories:
            data_yml_content += f"  {category['id']}: {category['name']}\n"

        with open(os.path.join(root_target, 'data.yml'), 'w') as f:
            f.write(data_yml_content)


        # Now create annotations folder in root_target
