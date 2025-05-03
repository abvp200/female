import os
import sys
import random

sys.path.append("/mnt/female-data-eval/")
sys.path.append("/mnt/female-data-eval/d_training")

from pathlib import Path
import torch
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from torch.utils.data import Subset
from tqdm.auto import tqdm

from d_training.faster_rcnn_resnet50_v7_strat import FasterRCNNModule
from coco_utils import get_coco_pytorchtransforms as get_coco
from transforms import ImageClassificationWithBBoxes2

# COCO-inspired distinct colors for classes
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25),
    (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60)
]

# Mapping from label index to name
class_map = {
    0: 'Adhesions.Dense',
    1: 'Adhesions.Filmy',
    2: 'Deep Endometriosis',
    3: 'Ovarian.Chocolate Fluid',
    4: 'Ovarian.Endometrioma',
    5: 'Superficial.Black',
    6: 'Superficial.Red',
    7: 'Superficial.Subtle',
    8: 'Superficial.White'
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

# Draw bounding boxes (no labels) with distinct colors
def draw_boxes(image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor, palette: list, width: int = 2) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for box, cls in zip(boxes.tolist(), labels.tolist()):
        color = palette[int(cls) % len(palette)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return image

# Annotate top-left corner with text lines (white font)
def annotate(image: Image.Image, lines: list, font: ImageFont.ImageFont=None) -> Image.Image:
    draw = ImageDraw.Draw(image)
    if font is None:
        try:
            font = ImageFont.truetype('arial.ttf', size=24)
        except:
            font = ImageFont.load_default()
    # compute total height
    padding = 4
    line_heights = [font.getsize(line)[1] for line in lines]
    total_h = sum(line_heights) + padding * (len(lines)+1)
    # background box
    draw.rectangle([0, 0, image.width, total_h], fill=(0, 0, 0))
    y = padding
    for line, h in zip(lines, line_heights):
        draw.text((padding, y), line, fill=(255,255,255), font=font)
        y += h + padding
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
    # Process multiple repeats
    for repeat_id in tqdm(range(5), desc='processing splits'):
        base_data_dir = (
            '/mnt/data_vilen/05_model_input/repeated_slim_stratified_train_val_test_split/'
            f'yolo_v6_all_classes_COCO_v4_noEmpty/repeated_split_{repeat_id}'
        )
        model_checkpoint = (
            f'/mnt/data/06_models/resnet50_FINAL_v3_v7_55_repeat{repeat_id}/checkpoints/chkpnt_FINAL'
        )
        output_dir = (
            f'/mnt/data/06_models/resnet50_FINAL_v3_v7_55_repeat{repeat_id}/vis'
        )

        args = Arguments(
            data_dir=base_data_dir,
            checkpoint=model_checkpoint,
            split='train',
            num_images=1000,
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
        combined_dir = output_base / args.split / 'gt_and_preds_combined'
        for d in [gt_dir, pred_dir, combined_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Seed and dataset
        random.seed(args.seed)
        dataset = get_dataset(args.data_dir, args.split)
        base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        selected = indices[:args.num_images]

        # Locate images
        candidate_dirs = [
            Path(args.data_dir)/'images',
            Path(args.data_dir)/args.split/'images',
            Path(args.data_dir)/'images'/args.split,
            Path(args.data_dir)/args.split,
            Path(args.data_dir)
        ]
        for d in candidate_dirs:
            if d.exists():
                image_dir = d; break

        # Load model
        model = FasterRCNNModule.load_from_checkpoint(
            args.checkpoint, num_classes=len(class_map)
        )
        model.eval().to(args.device)

        count = 0
        for idx in tqdm(selected, desc='images'):
            ds_idx = dataset.indices[idx] if isinstance(dataset, Subset) else idx
            img_info = base_ds.coco.imgs[base_ds.ids[ds_idx]]
            stem = Path(img_info['file_name']).stem
            img_path = image_dir/ img_info['file_name']
            original = Image.open(img_path).convert('RGB')
            resized = original.resize((1920,1080))

            # Ground truth boxes
            ann_ids = base_ds.coco.getAnnIds(imgIds=[base_ds.ids[ds_idx]], iscrowd=None)
            anns = base_ds.coco.loadAnns(ann_ids)
            raw_boxes, raw_labels = [], []
            for ann in anns:
                x,y,w,h = ann['bbox']
                raw_boxes.append([x,y,x+w,y+h]); 
                raw_labels.append(ann['category_id'])
            gt_image = resized.copy()
            if raw_boxes:
                b = torch.tensor(raw_boxes)
                l = torch.tensor(raw_labels)
                scale_w, scale_h = 1920/original.width, 1080/original.height
                b[:,[0,2]]*=int(scale_w) 
                b[:,[1,3]]*=int(scale_h)
                gt_image = draw_boxes(gt_image, b, l, PALETTE)
            # Annotate top-left: filename + classes
            gt_lines = [stem] + [class_map[int(c)] for c in sorted(set(raw_labels))]
            gt_image = annotate(gt_image, gt_lines)
            gt_image.save(gt_dir/f"{stem}_gt.jpg")

            # Predictions
            inp = TF.to_tensor(resized).unsqueeze(0).to(args.device)
            with torch.no_grad(): pred = model.model(inp)[0]
            labels = pred['labels'].cpu().tolist()
            boxes = pred['boxes'].cpu()
            scores = pred['scores'].cpu().tolist()
            pred_image = draw_boxes(resized.copy(), boxes, torch.tensor(labels), PALETTE)
            pred_lines = [stem] + [f"{class_map[int(c)]}:{scores[i]:.1f}" for i,c in enumerate(labels)]
            pred_image = annotate(pred_image, pred_lines)
            pred_image.save(pred_dir/f"{stem}_pred.jpg")

            # Combined
            combined = Image.new('RGB',(1920,2160))
            combined.paste(gt_image,(0,0)); combined.paste(pred_image,(0,1080))
            combined.save(combined_dir/f"{stem}_combined.jpg")

            count+=1
        print(f"Saved {count} images for split {args.split} in {output_base/args.split}")

if __name__=='__main__':
    main()
