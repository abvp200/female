import os
import torch
import mlflow
import tempfile
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from fasterrcnn_cocodataset import CocoDataset, ComposeTransforms, ToTensor, Resize, Normalize
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import default_collate
from git import Repo
from git_utils import get_git_hash, create_git_tag
# from coco_utils import get_coco
from coco_utils import get_coco_pytorchtransforms as get_coco
import json
from torchvision.models import ResNet50_Weights
from transforms import ImageClassificationWithBBoxes2

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

def draw_and_save_images(images, targets, type, batch_index, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    for idx, (image, target) in enumerate(zip(images, targets)):
        # Denormalize image
        image = image * std + mean  # Apply the reverse of the normalization
        image = torch.clamp(image, 0, 1)  # Ensure all values are within the correct range

        # Convert tensor to PIL Image
        image = Image.fromarray((image.mul(255).permute(1, 2, 0).byte().numpy()))
        draw = ImageDraw.Draw(image)
        boxes = target['boxes']
        labels = target['labels']

        # Draw each bounding box
        for box, label in zip(boxes, labels):
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            draw.text((box[0], box[1]), str(label.item()), fill="red")

        # Save the image
        image.save(f'/mnt/{type}_batch_{batch_index}_image_{idx}.jpg')

# from torchvision import prototype as P




def collate_fn(batch):
    return tuple(zip(*batch))

RESOLUTION = 640

def get_transforms():
    return ImageClassificationWithBBoxes2(
        crop_size=None, 
        resize_size=RESOLUTION,
        class_mapping={ # 0 - Background
            0:1,
            1:2,
            2:3,
            3:4,
            4:5,
            5:5,
            6:6,
            7:7,
            8:8,
            9:9
    })

    # return P.models.ResNet50_Weights.IMAGENET1K_V2.transforms()

class CocoDataModule(LightningModule):
    def __init__(self, data_dir, batch_size=4, take_subset=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.take_subset = take_subset  # 0 - means full dataset is used, 100 - means only 100 images will be taken

    def process_and_save_train_batches(self):
        dataloader = self.train_dataloader()
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Stop after N batches
                break
            images, targets = batch
            draw_and_save_images(images, targets, 'train', i)

    def setup(self, stage=None):
        self.transforms = get_transforms()

        self.train_dataset = get_coco(root=self.data_dir, 
                                      anno_file_path=os.path.join(self.data_dir, 'annotations/instances_train.json'),
                                      image_set='train', 
                                      transforms=self.transforms, 
                                      mode="instances", 
                                      use_v2=False, 
                                      with_masks=False, 
                                      take_subset=self.take_subset)

        self.val_dataset = get_coco(root=self.data_dir, 
                                      anno_file_path=os.path.join(self.data_dir, 'annotations/instances_val.json'),
                                      image_set='val', 
                                      transforms=self.transforms, 
                                      mode="instances", 
                                      use_v2=False, 
                                      with_masks=False, 
                                      take_subset=self.take_subset)
        
        self.test_dataset = get_coco(root=self.data_dir, 
                                      anno_file_path=os.path.join(self.data_dir, 'annotations/instances_test.json'),
                                      image_set='test', 
                                      transforms=self.transforms, 
                                      mode="instances", 
                                      use_v2=False, 
                                      with_masks=False, 
                                      take_subset=self.take_subset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

class FasterRCNNModule(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        # Load a FasterRCNN model with a ResNet-101 FPN backbone
        backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.IMAGENET1K_V2, trainable_layers=3)
        self.model = FasterRCNN(backbone, num_classes=num_classes)

        # weights = P.models.ResNet50_Weights.IMAGENET1K_V2
        # self.model = P.models.resnet50(weights=weights)

        # Replace the classifier with a new one, for fine-tuning
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        # loss = torch.sum(torch.stack([torch.sum(l['scores']) for l in loss_dict])).item()  #
        loss = sum(loss for loss in loss_dict.values())
        # self.log('train_loss', loss.item(), batch_size=len(images), sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(images), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = torch.sum(torch.stack([torch.sum(l['scores']) for l in loss_dict])).item()
        # self.log('val_loss', loss, batch_size=len(images), sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, logger=True, batch_size=len(images), sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer




if __name__ == "__main__":
    import mlflow
    
    # mlflow_tracking_uri = "http://mlflow-server:5000"

    version = 'v14'
    resolution = f'{RESOLUTION}x{RESOLUTION}'
    epochs = 1
    
    # mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    for i in [0]:  #,1,2,3,4]:  #
        # Main script
        data_dir = f'/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/yolo_v6_all_classes_COCO_v4_noEmpty/repeated_split_{i}'
        num_classes = 9
        split_type = 'StratSplit'

        # # Configure MLFlow logger
        # mlflow_logger = MLFlowLogger(
        #     experiment_name=f'FasterRCNN_ResNet50_Img32_ALL_CLS_DSet_{version}__{split_type}_classesNo{num_classes}_ep{epochs}_{resolution}',
        #     tracking_uri='http://mlflow-server:5000'
        # )

        # # Git tagging
        # repo = Repo(os.getcwd())
        # tag_name = f"experiment-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        # tag = create_git_tag(repo, tag_name)

        # if tag:
        #     tag_hash = get_git_hash(repo)
        #     print(f"Created Git tag: {tag_name} with hash: {tag_hash}")

        # Initialize your data module and model
        coco_data = CocoDataModule(data_dir)
        coco_data.setup()  # Explicitly call setup if necessary
        coco_data.process_and_save_train_batches()
        # faster_rcnn = FasterRCNNModule(num_classes)


        # dataset_metadata = {
        #     "train_dataset": {
        #         "path": os.path.join(data_dir, 'annotations/instances_train.json'),
        #         "num_samples": len(coco_data.train_dataset),
        #         "transforms": str(coco_data.transforms)
        #     },
        #     "val_dataset": {
        #         "path": os.path.join(data_dir, 'annotations/instances_val.json'),
        #         "num_samples": len(coco_data.val_dataset),
        #         "transforms": str(coco_data.transforms)
        #     },
        #     "test_dataset": {
        #         "path": os.path.join(data_dir, 'annotations/instances_test.json'),
        #         "num_samples": len(coco_data.test_dataset),
        #         "transforms": str(coco_data.transforms)
        #     }
        # }
        # # Specify the file name
        # file_name_dataset_metadata = "dataset_metadata.json"

        # # Save the dictionary to a JSON file
        # with open(file_name_dataset_metadata, "w") as json_file:
        #     json.dump(dataset_metadata, json_file, indent=4)


        # class_ids_str = 'allClasses'  #'_'.join([str(c) for c in class_ids])
        # model_name = f'resnet50_FINAL_v3_{version}_repeat{i}'

        
        # weights ='ResNet50_Weights.IMAGENET1K_V2'
        
        # checkpoint_name = f"{model_name}_{weights.replace('.','')}_{class_ids_str}_{version}_{epochs}ep_{resolution}_augm_{split_type}_repeat{i}"

        # train_params = dict(
        #     max_epochs=epochs,
        #     precision=32,
        #     # gradient_cli_val=gradient_cli_val,
        #     # enable_checkpointing=True,
        #     callbacks=[],  #[checkpoint_callback],
        #     logger=[mlflow_logger],
        #     auto_scale_batch_size=True,
        #     strategy="ddp",  # https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_expert.html?highlight=strategy#what-is-a-strategy
        #     accelerator="gpu", 
        #     devices=-1
        # )

        # trainer = pl.Trainer(**train_params)

        
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     path_dataset_metadata = Path(tmp_dir, file_name_dataset_metadata)
        #     path_dataset_metadata.write_text(json.dumps(dataset_metadata))

        #     mlflow_logger.experiment.log_artifact(run_id=mlflow_logger.run_id, local_path=path_dataset_metadata)
            
        #     # Train the model
        #     trainer.fit(faster_rcnn, datamodule=coco_data)

        #     trainer.save_checkpoint(os.path.join('/mnt', 'data', '06_models', model_name, 'checkpoints', 'chkpnt_FINAL'))
