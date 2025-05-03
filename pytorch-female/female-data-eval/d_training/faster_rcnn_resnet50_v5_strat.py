import os
import torch
import mlflow
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
import torchvision
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
from typing import List, Dict, Optional
# from torchvision import prototype as P


def collate_fn(batch):
    return tuple(zip(*batch))


RESOLUTION = '1920x1080'

def get_transforms():
    return ImageClassificationWithBBoxes2(
        crop_size=None, 
        resize_size=[1920,1080],
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

# class FasterRCNNModule(LightningModule):
#     def __init__(self, num_classes):
#         super().__init__()
#         # Load a FasterRCNN model with a ResNet-101 FPN backbone
#         backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.IMAGENET1K_V2, trainable_layers=3)
#         self.model = FasterRCNN(backbone, num_classes=num_classes)

#         # weights = P.models.ResNet50_Weights.IMAGENET1K_V2
#         # self.model = P.models.resnet50(weights=weights)

#         # Replace the classifier with a new one, for fine-tuning
#         in_features = self.model.roi_heads.box_predictor.cls_score.in_features
#         self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#     def forward(self, images, targets):
#         return self.model(images, targets)

#     def training_step(self, batch, batch_idx):
#         images, targets = batch
#         loss_dict = self.model(images, targets)
#         # loss = torch.sum(torch.stack([torch.sum(l['scores']) for l in loss_dict])).item()  #
#         loss = sum(loss for loss in loss_dict.values())
#         # self.log('train_loss', loss.item(), batch_size=len(images), sync_dist=True)
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(images), sync_dist=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         images, targets = batch
#         loss_dict = self.model(images, targets)
#         loss = torch.sum(torch.stack([torch.sum(l['scores']) for l in loss_dict])).item()
#         # self.log('val_loss', loss, batch_size=len(images), sync_dist=True)
#         self.log('val_loss', loss, prog_bar=True, logger=True, batch_size=len(images), sync_dist=True)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
#         return optimizer

ASSW_IOU_THRESHOLD = 0.5

nms_conf_thres: float = .5 #@param {type:"number"}
nms_iou_thres: float = ASSW_IOU_THRESHOLD #@param {type:"number"}
nms_max_det:int = 100 #@param {type:"integer"}
nms_agnostic_nms: bool = False #@param {type:"boolean"}

USE_NMS = True


def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    '''Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h].'''
    xx = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    
    # Convert
    xx[..., 2] = xx[..., 2] - xx[..., 0]  # width = x2 - x1
    xx[..., 3] = xx[..., 3] - xx[..., 1]  # height = y2 - y1
    xx[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x = (x1 + x2) / 2
    xx[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y = (y1 + y2) / 2
    
    return xx


def convert_to_nms_format(prediction: List[Dict]):
    '''
    We assume batch size is always 1 and label is only 1 - as we have per-class-classifiers
    '''
    boxes = prediction[0]['boxes'].to('cuda')
    scores = prediction[0]['scores'].to('cuda')
    
    boxes = xyxy2xywh(boxes)
    # xyxyx -> xywh
    # [x,y,x,yconf,class]
    return torch.cat([boxes.to('cuda'), 
                        scores.reshape(-1,1).to('cuda'),
                        torch.ones(boxes.shape[0], 1).to('cuda')],
                    axis=1).to('cuda')


def convert_from_nms_format(predictions_nms: torch.Tensor, predictions: List[Dict]):
    '''
    We assume batch size is always 1 and label is only 1 - as we have per-class-classifiers
    '''
    predictions[0]['boxes'] = predictions_nms[:, :4].to('cuda')
    predictions[0]['scores'] = predictions_nms[:, 5].to('cuda')
    # IMPORTANT! We assume single class only - thuse only '1'
    predictions[0]['labels'] = torch.ones(predictions_nms[:, 5].shape[0]).to('cuda')
    
    return predictions


def apply_nms(predictions: List[Dict]) -> List[Dict]:
    # convert predictions/targets to NMS format
    predictions_nms = convert_to_nms_format(predictions)
    # targets = convert_to_nms_format(targets)

    predictions_nms = predictions_nms.unsqueeze(0).to('cuda') # expand for batch dim
    classes:Optional[List[int]] = None # the classes to keep
    predictions_nms_processed = non_max_suppression(predictions_nms, nms_conf_thres, nms_iou_thres, classes, nms_agnostic_nms, max_det=nms_max_det)[0]

    return convert_from_nms_format(predictions_nms_processed, predictions)



import os
import time
import numpy as np
import cv2
import torch
import torchvision


# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads


def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = prediction.shape[2] - 5  # number of classes
    pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres, torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres)  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break  # time limit exceeded

    return output

from torchmetrics.detection.mean_ap import MeanAveragePrecision

class FasterRCNNModule(LightningModule):
    def __init__(self, num_classes, dataset: torch.utils.data.Dataset, fine_tune=False, eval_only=False, metric_step=10):
        super().__init__()
        self.dataset = dataset
        self.global_metric = MeanAveragePrecision(iou_thresholds=[ASSW_IOU_THRESHOLD], class_metrics=True)
        self.running_metric = MeanAveragePrecision(iou_thresholds=[ASSW_IOU_THRESHOLD], class_metrics=True)
        self.metric_step = metric_step
        self.map_per_class = []
        self.mar_per_class = []
        self.running_cls = set()
        self.running_idx = None
        self.global_idx = None

        self.model = self.build_model(num_classes, fine_tune)

        if not eval_only:
            self.params = [p for p in self.model.parameters() if p.requires_grad]
            self.model.train()  # put model into training state by default
        else:
            self.model.eval()

    def build_model(self, num_classes: int, fine_tune: bool):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    def forward(self, images, targets=None):
        loss_dict = self.model(images, targets)
        if targets is not None:
            losses = sum(loss for loss in loss_dict.values())
            return losses
        else:
            return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.params, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return optimizer

    def calc_model_loss(self, batch, batch_indx):
        images, targets = batch
        self.model.train()
        loss_dict_list = self.model(images, targets)
        losses_sum = sum(loss for loss in loss_dict_list.values())
        losses_dict = {k: v.item() for k, v in loss_dict_list.items()}
        return losses_sum, losses_dict

    def training_step(self, batch, batch_indx):
        losses_sum, losses_dict = self.calc_model_loss(batch, batch_indx)
        self.log("train_loss", losses_sum.item())
        for k, v in losses_dict.items():
            self.log(f"Train/{k}", v)
        return losses_sum

    def validation_step(self, batch, batch_indx):
        loss, losses_dict = self.calc_model_loss(batch, batch_indx)
        self.log("val_loss", loss.item())
        for k, v in losses_dict.items():
            self.log(f"Validation/{k}", v)
        return loss

    def test_step(self, batch, batch_indx):
        images, targets = batch
        self.model.eval()
        predictions = self.model(images)
        
        if USE_NMS:
            predictions = apply_nms(predictions)
        
        self.global_metric.update(predictions, targets)
        self.update_metric(predictions, targets, batch_indx)
        
        loss, losses_dict = self.calc_model_loss(batch, batch_indx)
        self.log("test_loss", loss.item())
        for k, v in losses_dict.items():
            self.log(f"Test/{k}", v)
        
        return None

    def update_metric(self, predictions, targets, idx):
        video_name, frame = self.dataset.image_idx_2_image_name[idx].split('__')
        
        if USE_NMS:
            predictions = apply_nms(predictions)
            
        for t in targets:
            for key, value in t.items():
                if isinstance(value, torch.Tensor):
                    t[key] = value.to('cuda')

        if self.global_idx is None and self.running_idx is None:
            self.global_idx = video_name
            self.running_idx = int(frame)

        if self.global_idx != video_name or abs(self.running_idx - int(frame)) > self.metric_step:
            metrics = self.running_metric.compute()
            map_per_class_final = np.full((10,), np.nan)
            mar_per_class_final = np.full((10,), np.nan)

            map_per_class = metrics['map_per_class'].numpy()
            map_running_cls = np.array(list(self.running_cls))
            map_running_cls = map_running_cls[map_per_class != -1]
            map_per_class_final[map_running_cls] = map_per_class[map_per_class != -1]

            mar_per_class = metrics['mar_100_per_class'].numpy()
            mar_running_cls = np.array(list(self.running_cls))
            mar_running_cls = mar_running_cls[mar_per_class != -1]
            mar_per_class_final[mar_running_cls] = mar_per_class[mar_per_class != -1]

            self.map_per_class.append(map_per_class_final)
            self.mar_per_class.append(mar_per_class_final)
            self.running_metric = MeanAveragePrecision(iou_thresholds=[ASSW_IOU_THRESHOLD], class_metrics=True)
            self.global_idx = video_name
            self.running_idx = int(frame)
            self.running_cls = set()

        self.running_cls |= set(targets[0]['labels'].cpu().numpy().tolist())
        self.running_cls |= set(predictions[0]['labels'].cpu().numpy().tolist())
        self.running_metric.update(predictions, targets)

    def __custom_histogram_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def training_epoch_end(self, outputs):
        self.__custom_histogram_adder()
        avg_loss = -1
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            loss = sum(output['loss'].mean() for output in gathered) / len(outputs)
            avg_loss = loss.item()
        
        self.logger.experiment.add_scalar("Loss/Train per Epoch", avg_loss, self.current_epoch)

    def validation_epoch_end(self, outputs):
        if len(outputs) < 5:
            return
        avg_loss = -1
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            stacked_gathered = torch.hstack(gathered)
            length = stacked_gathered.shape[0]
            loss = stacked_gathered.mean().item() / length
            avg_loss = loss
        
        self.logger.experiment.add_scalar("Loss/Validation per Epoch", avg_loss, self.current_epoch)


if __name__ == "__main__":
    import mlflow
    
    mlflow_tracking_uri = "http://mlflow-server:5000"

    version = 'v20'
    resolution = f'{RESOLUTION}'  #x{RESOLUTION}'
    epochs = 10
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    for i in [0]:  # ,1,2,3,4
        # Main script
        data_dir = f'/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/yolo_v6_all_classes_COCO_v4_noEmpty/repeated_split_{i}'
        num_classes = 9
        split_type = 'StratSplit'

        # Configure MLFlow logger
        mlflow_logger = MLFlowLogger(
            experiment_name=f'FasterRCNN_ResNet50_ALL_CLS_DSet_{version}__{split_type}_classesNo{num_classes}_ep{epochs}_{resolution}',
            tracking_uri='http://mlflow-server:5000'
        )

        # Git tagging
        repo = Repo(os.getcwd())
        tag_name = f"experiment-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        tag = create_git_tag(repo, tag_name)

        if tag:
            tag_hash = get_git_hash(repo)
            print(f"Created Git tag: {tag_name} with hash: {tag_hash}")

        # Initialize your data module and model
        coco_data = CocoDataModule(data_dir)
        coco_data.setup()  # Explicitly call setup if necessary

        faster_rcnn = FasterRCNNModule(num_classes=num_classes, 
                                       dataset=train, fine_tune=False, eval_only=False, metric_step=10)


        dataset_metadata = {
            "train_dataset": {
                "path": os.path.join(data_dir, 'annotations/instances_train.json'),
                "num_samples": len(coco_data.train_dataset),
                "transforms": str(coco_data.transforms)
            },
            "val_dataset": {
                "path": os.path.join(data_dir, 'annotations/instances_val.json'),
                "num_samples": len(coco_data.val_dataset),
                "transforms": str(coco_data.transforms)
            },
            "test_dataset": {
                "path": os.path.join(data_dir, 'annotations/instances_test.json'),
                "num_samples": len(coco_data.test_dataset),
                "transforms": str(coco_data.transforms)
            }
        }
        # Specify the file name
        file_name_dataset_metadata = "dataset_metadata.json"

        # Save the dictionary to a JSON file
        with open(file_name_dataset_metadata, "w") as json_file:
            json.dump(dataset_metadata, json_file, indent=4)


        class_ids_str = 'allClasses'  #'_'.join([str(c) for c in class_ids])
        model_name = f'resnet50_FINAL_v4_{version}_repeat{i}'

        
        weights ='ResNet50_Weights.IMAGENET1K_V2'
        
        checkpoint_name = f"{model_name}_{weights.replace('.','')}_{class_ids_str}_{version}_{epochs}ep_{resolution}_augm_{split_type}_repeat{i}"

        train_params = dict(
            max_epochs=epochs,
            precision=32,
            # gradient_cli_val=gradient_cli_val,
            # enable_checkpointing=True,
            callbacks=[],  #[checkpoint_callback],
            logger=[mlflow_logger],
            auto_scale_batch_size=True,
            strategy="ddp",  # https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_expert.html?highlight=strategy#what-is-a-strategy
            accelerator="gpu", 
            devices=-1
        )

        trainer = pl.Trainer(**train_params)

        
        with tempfile.TemporaryDirectory() as tmp_dir:
            path_dataset_metadata = Path(tmp_dir, file_name_dataset_metadata)
            path_dataset_metadata.write_text(json.dumps(dataset_metadata))

            mlflow_logger.experiment.log_artifact(run_id=mlflow_logger.run_id, local_path=path_dataset_metadata)
            
            # Train the model
            trainer.fit(faster_rcnn, datamodule=coco_data)

            trainer.save_checkpoint(os.path.join('/mnt', 'data', '06_models', model_name, 'checkpoints', 'chkpnt_FINAL'))
