from collections import defaultdict
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset


from torchvision.transforms import functional as F

class ComposeTransforms(object):
    """Custom transforms to apply to both images and targets."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        
        image_t = []
        target_t = []
        for transform in self.transforms:
            image, target = transform(image, target)
            image_t.append(image)
            target_t.append(target)
        return image_t, target_t

class ToTensor(object):
    """Convert PIL Images and targets to tensors."""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Resize(object):
    """Resize the input PIL Image to the given size and adjust bounding boxes accordingly."""
    def __init__(self, size):
        self.size = size  # This should be a tuple (new_height, new_width)

    def __call__(self, image, target):
        # Capture original dimensions before resizing
        orig_width, orig_height = image.shape[2], image.shape[1]  #image.size
        
        # Resize image
        image = F.resize(image, (orig_width, orig_height))  #self.size)
        
        # Adjust bounding boxes
        if 'boxes' in target:
            # Calculate scale factors
            width_scale = self.size[1] / orig_width
            height_scale = self.size[0] / orig_height
            
            # Resize bounding boxes
            boxes = target['boxes']
            # boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = boxes.clone().detach().float()
            
            boxes[:, [0, 2]] *= width_scale  # Scale x coordinates
            boxes[:, [1, 3]] *= height_scale  # Scale y coordinates
            target['boxes'] = boxes

        return image, target

# class Resize(object):
#     """Resize the input PIL Image to the given size."""
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, image, target):
#         image = F.resize(image, self.size)
#         target['boxes'] = torch.tensor(target['boxes']) * (self.size[0] / image.width)  # Example scaling factor
#         return image, target

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target



class CocoDataset(Dataset):
    """PyTorch dataset for COCO annotations."""

    def __init__(self, data_dir, anno_file_path, transforms=None):
        """Load COCO annotation data."""
        self.data_dir = Path(data_dir)
        self.transforms = transforms

        # load the COCO annotations json
        # anno_file_path = self.data_dir/'coco_annotations.json'
        with open(str(anno_file_path)) as file_obj:
            self.coco_data = json.load(file_obj)
        # put all of the annos into a dict where keys are image IDs to speed up retrieval
        self.image_id_to_annos = defaultdict(list)
        for anno in self.coco_data['annotations']:
            image_id = anno['image_id']
            self.image_id_to_annos[image_id] += [anno]

    def __len__(self):
        return len(self.coco_data['images'])

    def __getitem__(self, index):
        """Return tuple of image and labels as torch tensors."""
        image_data = self.coco_data['images'][index]
        image_id = image_data['id']
        image_path = self.data_dir/'images'/image_data['file_name']
        image = Image.open(image_path)

        annos = self.image_id_to_annos[image_id]
        anno_data = {
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': [],
        }
        for anno in annos:
            coco_bbox = anno['bbox']
            left = coco_bbox[0]
            top = coco_bbox[1]
            right = coco_bbox[0] + coco_bbox[2]
            bottom = coco_bbox[1] + coco_bbox[3]
            area = coco_bbox[2] * coco_bbox[3]
            anno_data['boxes'].append([left, top, right, bottom])
            anno_data['labels'].append(anno['category_id'])
            anno_data['area'].append(area)
            anno_data['iscrowd'].append(anno['iscrowd'])

        target = {
            'boxes': torch.as_tensor(anno_data['boxes'], dtype=torch.float32),
            'labels': torch.as_tensor(anno_data['labels'], dtype=torch.int64),
            'image_id': torch.tensor([image_id]),  # pylint: disable=not-callable (false alarm)
            'area': torch.as_tensor(anno_data['area'], dtype=torch.float32),
            'iscrowd': torch.as_tensor(anno_data['iscrowd'], dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target