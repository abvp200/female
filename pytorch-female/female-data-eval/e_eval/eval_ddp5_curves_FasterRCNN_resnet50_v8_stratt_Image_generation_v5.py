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

from tqdm.auto import tqdm

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
    for i in tqdm([0,1,2,3,4], desc='processing splits'): # 
        repeat_id = i
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
            split='train',  # 'val' or 'train'/'test'
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
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        combined_dir.mkdir(parents=True, exist_ok=True)

        # Seed and dataset
        random.seed(args.seed)
        dataset = get_dataset(args.data_dir, args.split, take_subset=0)
        # If dataset is a Subset wrapper, get the underlying
        base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        selected = indices  #[:args.num_images]

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

        # num_saved_images = 0

        # count = 0
        for idx in tqdm(selected, desc='processing dataset images'):
            # Map subset index
            ds_idx = dataset.indices[idx] if isinstance(dataset, Subset) else idx
            img_info = base_ds.coco.imgs[base_ds.ids[ds_idx]]
            img_path = image_dir / img_info['file_name']
            original = Image.open(img_path).convert('RGB')
            resized = original.resize((1920, 1080))

            # Use original filename stem for saving
            stem = Path(img_info['file_name']).stem

            # Ground truth
            image_id = base_ds.ids[ds_idx]
            ann_ids = base_ds.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
            anns = base_ds.coco.loadAnns(ann_ids)
            raw_boxes, raw_labels = [], []
            for ann in anns:
                x,y,w,h = ann['bbox']
                raw_boxes.append([x, y, x+w, y+h])
                raw_labels.append(ann['category_id'])
            if raw_boxes:
                boxes_tensor = torch.tensor(raw_boxes)
                labels_tensor = torch.tensor(raw_labels)
                scale_w = 1920/original.width
                scale_h = 1080/original.height
                boxes_scaled = boxes_tensor.clone()
                boxes_scaled[:,[0,2]]*=int(scale_w)
                boxes_scaled[:,[1,3]]*=int(scale_h)
                gt_img = draw_boxes(resized.copy(), boxes_scaled, labels_tensor, class_map)
            else:
                gt_img = resized.copy()


            # Prediction: model inference on resized image
            input_tensor = TF.to_tensor(resized).unsqueeze(0).to(args.device)
            with torch.no_grad():
                pred = model.model(input_tensor)[0]


            if pred['boxes'].cpu().shape[0] > 0:
                
                gt_img.save(gt_dir/ f"{stem}_gt.jpg")
                
                pred_img = draw_boxes(resized.copy(), pred['boxes'].cpu(), pred['labels'].cpu(), class_map)
                pred_img.save(pred_dir/ f"{stem}_pred.jpg")
                
                # Combined image (stack GT above prediction)
                combined = Image.new('RGB', (1920, 1080*2))
                combined.paste(gt_img, (0,0))
                combined.paste(pred_img, (0,1080))
                combined.save(combined_dir/ f"{stem}_combined.jpg")
                
                # num_saved_images = num_saved_images + 1
                
                # if num_saved_images >= args.num_images:
                #     break

        print(f"Saved {len(selected)} GT/pred pairs under {output_base / args.split}")


if __name__ == '__main__':
    main()
