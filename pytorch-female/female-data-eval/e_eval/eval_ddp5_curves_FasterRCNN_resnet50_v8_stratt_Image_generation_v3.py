import os
import sys

sys.path.append("/mnt/female-data-eval/")
sys.path.append("/mnt/female-data-eval/d_training")

import random
from pathlib import Path

import torch
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from torch.utils.data import Subset

from d_training.faster_rcnn_resnet50_v7_strat import FasterRCNNModule
from coco_utils import get_coco_pytorchtransforms as get_coco
from transforms import ImageClassificationWithBBoxes2

# Mapping from COCO label indices to class names
class_map = {
    1: 'Adhesions.Dense',
    2: 'Adhesions.Filmy',
    3: 'Deep Endometriosis',
    4: 'Ovarian.Chocolate Fluid',
    5: 'Ovarian.Endometrioma',
    6: 'Superficial.Black',
    7: 'Superficial.Red',
    8: 'Superficial.Subtle',
    9: 'Superficial.White'
}

# Default transforms (resize to 1920x1080)
def get_transforms():
    return ImageClassificationWithBBoxes2(
        crop_size=None,
        resize_size=[1920, 1080],
        class_mapping={i: i for i in range(10)}
    )

# Build COCO-style dataset
def get_dataset(data_dir: str, split: str, take_subset: int = 0):
    transforms = get_transforms()
    ann_file = os.path.join(data_dir, 'annotations', f'instances_{split}.json')
    return get_coco(
        root=data_dir,
        anno_file_path=ann_file,
        image_set=split,
        transforms=transforms,
        mode="instances",
        use_v2=False,
        with_masks=False,
        take_subset=take_subset
    )

# Draw bounding boxes + labels on PIL image
def draw_boxes(image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor, names: dict,
               outline_color=(255, 0, 0), width: int = 2) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, cls in zip(boxes.tolist(), labels.tolist()):
        x1, y1, x2, y2 = box
        label = names.get(int(cls), str(int(cls)))
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=width)
        text_w, text_h = font.getsize(label)
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=outline_color)
        draw.text((x1, y1 - text_h), label, fill=(255, 255, 255), font=font)
    return image

class Arguments(BaseModel):
    data_dir: str
    checkpoint: str
    split: str
    num_images: int
    seed: int
    device: str
    output_dir: str


def main():
    # Configuration
    repeat_id = 0
    base_data_dir = (
        '/mnt/data_vilen/05_model_input/repeated_slim_stratified_train_val_test_split/'
        'yolo_v6_all_classes_COCO_v4_noEmpty/repeated_split_{}'.format(repeat_id)
    )
    model_checkpoint = (
        '/mnt/data/06_models/resnet50_FINAL_v3_v7_55_repeat{}/checkpoints/chkpnt_FINAL'.
        format(repeat_id)
    )
    output_dir = (
        '/mnt/data/06_models/resnet50_FINAL_v3_v7_55_repeat{}/vis'.
        format(repeat_id)
    )

    # Parse/validate args
    args = Arguments(
        data_dir=base_data_dir,
        checkpoint=model_checkpoint,
        split='val',  # or 'train'/'test'
        num_images=100,
        seed=42,
        device='cuda:1' if torch.cuda.is_available() else 'cpu',
        output_dir=output_dir
    )

    # Prepare output dirs
    output_base = Path(args.output_dir)
    if output_base.is_file():
        output_base = output_base.parent
    gt_dir = output_base / args.split / 'ground_truth'
    pred_dir = output_base / args.split / 'predictions'
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Seed and dataset
    random.seed(args.seed)
    dataset = get_dataset(args.data_dir, args.split, take_subset=0)
    # If dataset is a Subset wrapper, get the underlying
    base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected = indices[:args.num_images]

    # Locate raw images directory
    candidate_dirs = [
        Path(args.data_dir) / 'images',
        Path(args.data_dir) / args.split / 'images',
        Path(args.data_dir) / 'images' / args.split,
        Path(args.data_dir) / args.split,
        Path(args.data_dir)
    ]
    for d in candidate_dirs:
        if d.exists() and d.is_dir():
            image_dir = d
            break
    else:
        raise FileNotFoundError(f"No image folder found under {args.data_dir}")

    # Load model
    model = FasterRCNNModule.load_from_checkpoint(
        args.checkpoint,
        num_classes=len(class_map)
    )
    model.eval().to(args.device)

    # Iterate selection
    for idx in selected:
        # Get annotation image id
        ds_idx = idx
        # If Subset, map through indices
        if isinstance(dataset, Subset):
            ds_idx = dataset.indices[idx]
        img_info = base_ds.coco.imgs[base_ds.ids[ds_idx]]
        img_path = image_dir / img_info['file_name']
        original = Image.open(img_path).convert('RGB')

        # Ground truth
        _, target = dataset[idx]
        gt_img = draw_boxes(original.copy(), target['boxes'], target['labels'], class_map)
        gt_img.save(gt_dir / f"{idx:04d}_gt.jpg")

        # Prediction
        resized = original.resize((1920, 1080))
        input_tensor = TF.to_tensor(resized).unsqueeze(0).to(args.device)
        with torch.no_grad():
            pred = model.model(input_tensor)[0]
        pred_img = draw_boxes(original.copy(), pred['boxes'].cpu(), pred['labels'].cpu(), class_map)
        pred_img.save(pred_dir / f"{idx:04d}_pred.jpg")

    print(f"Saved {len(selected)} GT/pred pairs under {output_base / args.split}")

if __name__ == '__main__':
    main()
