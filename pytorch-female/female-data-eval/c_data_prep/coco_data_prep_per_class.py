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
    5: 4,  # Category 5 merged into Category 4
    6: 5,
    7: 6,
    8: 7,
    9: 8
}

# Function to read annotations from JSON file
def read_json(json_file_path: str):
    """
    Read and load JSON data from a file.

    Args:
        json_file_path (str): The path to the JSON file to be read.

    Returns:
        dict: A dictionary containing the JSON data loaded from the file.
    """
    with open(json_file_path, 'r') as f:
        json_dict = json.load(f)
    return json_dict


def create_folder(folder_path):
    # Remove the folder if it exists
    if os.path.exists(folder_path):
        rmtree(folder_path)
    
    # Create the folder
    os.makedirs(folder_path)


def create_symbolic_links(source_dir, output_dir, filter_images=None):
    """
    Create target output_dir filled in with symbolic links to all the files in source_dir.

    Args:
        source_dir (str): The directory containing the original files.
        output_dir (str): The directory where symbolic links will be created.
        filter_images (set): Set of image filenames to include in the symbolic links.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    for img_file in os.listdir(source_dir):
        if filter_images is None or img_file in filter_images:
            src_path = os.path.join(source_dir, img_file)
            dst_path = os.path.join(output_dir, img_file)
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)


def convert_annotations_for_split(coco_json_videos: Dict, split_dict: Dict[str, List[str]], source_img_folder: str, target_class_id: int):
    """
    Generate COCO annotations for images for a specific class.

    Args:
        coco_json_videos (dict): COCO video annotations.
        split_dict (dict): Split dictionary containing video names.
        source_img_folder (str): Source image folder path.
        target_class_id (int): Target class ID to filter annotations.

    Returns:
        dict: Dictionary with split annotations for the specific class.
    """

    def __remap_images(img_dicts_list: List[Dict[str, Any]], source_img_folder: str, old_image_id_to_new_image_id: Dict[str, int]):
        def map_img_anno(img_anno, new_id: int):
            return {
                "id": new_id,
                "width": img_anno['width'],
                "height": img_anno['height'],
                "file_name": os.path.join(source_img_folder, os.path.basename(img_anno['file_name'])),
                "license": img_anno.get('license', ''),
                "date_captured": img_anno.get('date_captured', '')
            }
        return [map_img_anno(img_anno, old_image_id_to_new_image_id[img_anno['id']]) for img_anno in img_dicts_list]

    def __image_id_is_in(image_id: str, video_names_list: List[str]) -> bool:
        return image_id.split('__')[0] in video_names_list

    def __remap_categories(categories: dict, original_class_id: int):
        remapped_categories = []

        for category in categories:
            if CATEGORY_MAPPING[category['id']] == original_class_id:
                remapped_category = copy.deepcopy(category)
                remapped_category['id'] = 0  # Set the category ID to 0 for the single-class dataset
                remapped_categories.append(remapped_category)

        return remapped_categories

    def __remap_annotations(annotations: List[dict], CATEGORY_MAPPING: dict, target_class_id: int):
        annotations_remapped = []
        image_id_counter = 0
        old_image_id_to_new_image_id = {}

        for item in annotations:
            mapped_class_id = CATEGORY_MAPPING[item['category_id']]
            if mapped_class_id == target_class_id:
                new_item = copy.deepcopy(item)
                new_item['category_id'] = 0  # Set the category ID to 0 for the single-class dataset

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
            CATEGORY_MAPPING, target_class_id
        )

        remapped_images = __remap_images(
            img_dicts_list=[i for i in images_anno if i['id'] in old_image_id_to_new_image_id],
            source_img_folder=source_img_folder,
            old_image_id_to_new_image_id=old_image_id_to_new_image_id
        )

        remapped_categories = __remap_categories(coco_json_videos['categories'], target_class_id)

        split_coco_annotations = {
            "images": remapped_images,
            "annotations": remapped_annotations,
            "categories": remapped_categories
        }

        splits_annotations[split_name] = split_coco_annotations

    return splits_annotations


def save_splits_annotations(splits_annotations, target_anno_subdir: str):
    create_folder(target_anno_subdir)

    for split, annotations in splits_annotations.items():
        file_name = f'instances_{split}.json'

        with open(os.path.join(target_anno_subdir, file_name), 'w') as json_file:
            json.dump(annotations, json_file)


def convert_annotations(coco_json_videos: Dict, split_dict_list: List[Dict[str, List[str]]], source_img_folder: str, target_dir: str):
    create_folder(target_dir)

    for class_id in set(CATEGORY_MAPPING.values()):
        class_target_dir = os.path.join(target_dir, f'class_{class_id}')
        create_folder(class_target_dir)

        for i, split_dict in tqdm(enumerate(split_dict_list), total=len(split_dict_list)):
            target_subdir = os.path.join(class_target_dir, f'repeated_split_{i}')
            target_img_subdir = os.path.join(target_subdir, 'images')
            target_anno_subdir = os.path.join(target_subdir, 'annotations')
            splits_annotations = convert_annotations_for_split(coco_json_videos, split_dict, source_img_folder, class_id)
            save_splits_annotations(splits_annotations, target_anno_subdir)

            # Gather image filenames to create symbolic links only for images in annotations
            image_filenames = {os.path.basename(img['file_name']) for split_anno in splits_annotations.values() for img in split_anno['images']}
            create_symbolic_links(source_img_folder, target_img_subdir, filter_images=image_filenames)


# Specify paths to dataset directories
images_dir = '/mnt/data_vilen/01_raw/BoundingBox_2022_10_FullUpdated/img2'
splits_json_path = "/mnt/data_vilen/02_intermediate/repeated_stratified_train_val_test_split/5_splits_train_val_test.json"
target_dir = '/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/yolo_v6_all_classes_COCO_per_class'
splits = read_json(splits_json_path)
coco_json = read_json('/mnt/data_vilen/02_intermediate/glenda_full_v5_bbox_and_segm.json')

# Split dataset and organize into train/validation/test sets
convert_annotations(coco_json, splits, images_dir, target_dir)
