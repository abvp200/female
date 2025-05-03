import os
import json
from tqdm.auto import tqdm
from shutil import rmtree
from typing import Dict, List, Any
import copy

CATEGORY_MAPPING = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8
}

# Function to read annotations from JSON file
def read_json(json_file_path: str):
    with open(json_file_path, 'r') as f:
        json_dict = json.load(f)
    return json_dict

def create_folder(folder_path):
    if os.path.exists(folder_path):
        rmtree(folder_path)
    os.makedirs(folder_path)

def create_symbolic_links(source_dir, output_dir, images_with_annotations):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating symbolic links for {len(images_with_annotations)} images.")
    for img_file in images_with_annotations:
        src_path = os.path.join(source_dir, img_file)
        dst_path = os.path.join(output_dir, img_file)
        print(f"Linking {src_path} to {dst_path}")
        if not os.path.exists(dst_path):
            os.symlink(src_path, dst_path)

def convert_annotations_for_split(coco_json_videos: Dict, split_dict: Dict[str, List[str]], source_img_folder: str):
    def __remap_images(img_dicts_list: List[Dict[str, Any]], source_img_folder: str, old_image_id_to_new_image_id: Dict[str, int]):
        def map_img_anno(img_anno, new_id: int):
            return {
                "id": new_id,
                "width": img_anno['width'],
                "height": img_anno['height'],
                "file_name": os.path.basename(img_anno['file_name']),
                "license": img_anno['license'],
                "date_captured": img_anno['date_captured']
            }
        remapped_images = [map_img_anno(img_anno, old_image_id_to_new_image_id[img_anno['id']])
                           for img_anno in img_dicts_list if img_anno['id'] in old_image_id_to_new_image_id]
        return remapped_images

    def __image_id_is_in(image_id: str, video_names_list: List[str]) -> bool:
        return image_id.split('__')[0] in video_names_list

    def __remap_categories(categories: dict):
        remapped_categories = []
        for category in categories:
            remapped_id = CATEGORY_MAPPING.get(category['id'], category['id'])
            category['id'] = remapped_id
            remapped_categories.append(category)
        return remapped_categories

    def __remap_annotations(annotations: List[dict], CATEGORY_MAPPING: dict):
        annotations_remapped = []
        image_id_counter = 0
        old_image_id_to_new_image_id = {}
        for item in annotations:
            if 'bbox' not in item or not item['bbox']:
                continue
            new_item = copy.deepcopy(item)
            remapped_id = CATEGORY_MAPPING.get(item['category_id'], item['category_id'])
            new_item['category_id'] = remapped_id
            old_image_id = new_item['image_id']
            if old_image_id not in old_image_id_to_new_image_id:
                old_image_id_to_new_image_id[old_image_id] = image_id_counter
                image_id_counter += 1
            new_item['image_id'] = old_image_id_to_new_image_id[old_image_id]
            new_item['boxes'] = new_item['bbox']
            annotations_remapped.append(new_item)
        return annotations_remapped, old_image_id_to_new_image_id

    splits_annotations = {}
    images_anno = [item for sublist in coco_json_videos['images'] for item in sublist]

    for split_name, split_videos_names in split_dict.items():
        remapped_annotations, old_image_id_to_new_image_id = __remap_annotations(
            [a for a in coco_json_videos['annotations'] if __image_id_is_in(a['image_id'], split_videos_names)],
            CATEGORY_MAPPING
        )

        remapped_images = __remap_images(
            img_dicts_list=[i for i in images_anno if __image_id_is_in(i['id'], split_videos_names)],
            source_img_folder=source_img_folder,
            old_image_id_to_new_image_id=old_image_id_to_new_image_id
        )

        remapped_categories = __remap_categories(coco_json_videos['categories'])

        split_coco_annotations = {
            "images": remapped_images,
            "annotations": remapped_annotations,
            "categories": remapped_categories
        }

        splits_annotations[split_name] = split_coco_annotations

    images_with_annotations = {img['file_name'] for split in splits_annotations.values() for img in split['images']}
    print(f"Total images with annotations: {len(images_with_annotations)}")
    return splits_annotations, images_with_annotations

def save_splits_annotations(splits_annotations, target_anno_subdir: str):
    create_folder(target_anno_subdir)
    for split, annotations in splits_annotations.items():
        file_name = f'instances_{split}.json'
        with open(os.path.join(target_anno_subdir, file_name), 'w') as json_file:
            json.dump(annotations, json_file)

def convert_annotations(coco_json_videos: Dict, split_dict_list: List[Dict[str, List[str]]], source_img_folder: str, target_dir: str):
    create_folder(target_dir)
    for i, split_dict in tqdm(enumerate(split_dict_list), total=len(split_dict_list)):
        target_subdir = os.path.join(target_dir, f'repeated_split_{i}')
        target_img_subdir = os.path.join(target_subdir, 'images')
        target_anno_subdir = os.path.join(target_subdir, 'annotations')
        splits_annotations, images_with_annotations = convert_annotations_for_split(coco_json_videos, split_dict, source_img_folder)
        save_splits_annotations(splits_annotations, target_anno_subdir)
        create_symbolic_links(source_img_folder, target_img_subdir, images_with_annotations)

# ################
# # create Stratified splits

# # Specify paths to dataset directories
# images_dir = '/mnt/data_vilen/01_raw/BoundingBox_2022_10_FullUpdated/img2'
# splits_json_path = "/mnt/data_vilen/02_intermediate/repeated_stratified_train_val_test_split/5_splits_train_val_test.json"
# target_dir = '/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/yolo_v6_all_classes_COCO_v4_noEmpty'
# splits = read_json(splits_json_path)
# coco_json = read_json('/mnt/data_vilen/02_intermediate/glenda_full_v5_bbox_and_segm.json')

# # Split dataset and organize into train/validation/test sets
# convert_annotations(coco_json, splits, images_dir, target_dir)

# ################
# # create nonStratified splits

# # Specify paths to dataset directories
# images_dir = '/mnt/data_vilen/01_raw/BoundingBox_2022_10_FullUpdated/img2'
# splits_json_path = "/mnt/data_vilen/02_intermediate/repeated_nonstratified_train_val_test_split/5_splits_train_val_test.json"
# target_dir = '/mnt/data_vilen/05_model_input/repeated_nonstratified_train_val_test_split/yolo_v6_all_classes_COCO_v4_noEmpty'
# splits = read_json(splits_json_path)
# coco_json = read_json('/mnt/data_vilen/02_intermediate/glenda_full_v5_bbox_and_segm.json')
# convert_annotations(coco_json, splits, images_dir, target_dir)

################
# create nonStratified splits2

# Specify paths to dataset directories
images_dir = '/mnt/data_vilen/01_raw/BoundingBox_2022_10_FullUpdated/img2'
splits_json_path = "/mnt/data_vilen/02_intermediate/repeated_nonstratified2_train_val_test_split/5_splits_train_val_test.json"
target_dir = '/mnt/data_vilen/05_model_input/repeated_nonstratified2_train_val_test_split/yolo_v6_all_classes_COCO_v4_noEmpty'
splits = read_json(splits_json_path)
coco_json = read_json('/mnt/data_vilen/02_intermediate/glenda_full_v5_bbox_and_segm.json')
convert_annotations(coco_json, splits, images_dir, target_dir)
