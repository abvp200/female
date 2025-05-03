import os
import sys

sys.path.append("/mnt/female-data-eval/")
sys.path.append("/mnt/female-data-eval/d_training")

import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from torchvision.transforms import functional as TF
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

# Default transforms for dataset (adapt to your pipeline)
def get_transforms():
    return ImageClassificationWithBBoxes2(
        crop_size=None,
        resize_size=[1920, 1080],
        class_mapping={i: i for i in range(10)}  # identity mapping
    )

# Build a COCO-style dataset for a given split
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

# Utility to draw boxes and labels on an image
def draw_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    names: dict,
    outline_color=(255, 0, 0),
    width: int = 2
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for b, cls in zip(boxes.tolist(), labels.tolist()):
        x1, y1, x2, y2 = b
        label = names.get(int(cls), str(int(cls)))
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=width)
        text_w, text_h = font.getsize(label)
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=outline_color)
        draw.text((x1, y1 - text_h), label, fill=(255, 255, 255), font=font)
    return image

class Arguments(BaseModel):
    data_dir: str    # Root directory containing COCO annotations and images
    checkpoint: str  # Path to the Faster R-CNN checkpoint (.ckpt)
    split: str       # Dataset split: 'train', 'val', or 'test'
    num_images: int  # Number of images to process
    seed: int        # Random seed for shuffling
    device: str      # Device for model inference
    output_dir: str  # Base directory to save visualizations


def main():
    # Hardcoded configurations
    repeat_id = 0
    base_data_dir = '/mnt/data_vilen/05_model_input/repeated_slim_stratified_train_val_test_split/yolo_v6_all_classes_COCO_v4_noEmpty/repeated_split_{}'.format(repeat_id)
    model_checkpoint = '/mnt/data/06_models/resnet50_FINAL_v3_v7_55_repeat{}/checkpoints/chkpnt_FINAL'.format(repeat_id)
    output_dir = '/mnt/data/06_models/resnet50_FINAL_v3_v7_55_repeat{}/vis'.format(repeat_id)

    # Instantiate arguments
    args = Arguments(
        data_dir=base_data_dir,
        checkpoint=model_checkpoint,
        split='train',
        num_images=10,
        seed=42,
        device='cuda:1' if torch.cuda.is_available() else 'cpu',
        output_dir=output_dir
    )

    # Resolve output base (ensure it's a directory)
    output_base = Path(args.output_dir)
    if output_base.is_file():
        output_base = output_base.parent
    # Define output subdirectories
    out_gt = output_base / args.split / 'ground_truth'
    out_pred = output_base / args.split / 'predictions'
    out_gt.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)

    # Set seed and load dataset
    random.seed(args.seed)
    dataset = get_dataset(args.data_dir, args.split, take_subset=0)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected = indices[: args.num_images]

    # Locate raw image directory
    candidates = [
        Path(args.data_dir) / 'images',
        Path(args.data_dir) / 'images' / args.split,
        Path(args.data_dir) / args.split / 'images',
        Path(args.data_dir) / args.split,
        Path(args.data_dir)
    ]
    for d in candidates:
        if d.exists() and d.is_dir():
            image_dir = d
            break
    else:
        raise FileNotFoundError(f"Cannot locate image directory under {args.data_dir}. Tried: {candidates}")

    # Load model
    model = FasterRCNNModule.load_from_checkpoint(
        args.checkpoint,
        num_classes=len(class_map)
    )
    model.eval().to(args.device)

    # Iterate and visualize
    for idx in selected:
        # Load raw image
        img_info = dataset.coco.imgs[dataset.ids[idx]]
        img_path = image_dir / img_info['file_name']
        original = Image.open(img_path).convert('RGB')

        # Draw ground truth
        _, target = dataset[idx]
        gt_img = draw_boxes(original.copy(), target['boxes'], target['labels'], class_map)
        gt_img.save(out_gt / f"{idx:04d}_gt.jpg")

        # Prepare input & predict
        resized = original.resize((1920, 1080))
        input_tensor = TF.to_tensor(resized).unsqueeze(0).to(args.device)
        with torch.no_grad():
            preds = model.model(input_tensor)[0]

        # Draw predictions
        pred_img = draw_boxes(original.copy(), preds['boxes'].cpu(), preds['labels'].cpu(), class_map)
        pred_img.save(out_pred / f"{idx:04d}_pred.jpg")

    print(f"Saved {len(selected)} GT/pred pairs to {output_base}/{args.split}")

if __name__ == "__main__":
    main()
