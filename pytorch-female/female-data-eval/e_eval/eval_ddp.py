import os
import sys
sys.path.append('/mnt/female-data-eval')
sys.path.append('/mnt/female-data-eval/d_training')

from tqdm.auto import tqdm
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from pycocotools.cocoeval import COCOeval
import json

from pathlib import Path
from d_training.faster_rcnn_resnet101 import CocoDataModule, FasterRCNNModule
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import MLFlowLogger


def delete_file(file_name):
    try:
        os.remove(file_name)
        print(f"File '{file_name}' deleted successfully.")
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    except PermissionError:
        print(f"Permission denied: unable to delete '{file_name}'.")
    except Exception as e:
        print(f"Error occurred while deleting file '{file_name}': {e}")


class EvaluationLightningModule(LightningModule):
    
    def __init__(self, 
                 model,
                 train_coco_results_json_fname='train_coco_results.json',
                 val_coco_results_json_fname='val_coco_results.json',
                 test_coco_results_json_fname='test_coco_results.json'):
        
        super().__init__()
        self.model = model
        self.train_coco_results = []
        self.train_coco_results_json_fname = train_coco_results_json_fname
        self.val_coco_results = []
        self.val_coco_results_json_fname = val_coco_results_json_fname
        self.test_coco_results = []
        self.test_coco_results_json_fname = test_coco_results_json_fname

    def training_step(self, batch, batch_idx):
        return self.evaluate_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.evaluate_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate_step(batch, batch_idx, "test")
    
    def evaluate_step(self, batch, batch_idx, stage):
        images, targets = batch
        outputs = self.model(images, targets)
        coco_results = []

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"])
            if len(output["boxes"]) > 0:
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box.tolist(),
                        "score": float(score)
                    })

        if stage == "train":
            self.train_coco_results.extend(coco_results)
        elif stage == "val":
            self.val_coco_results.extend(coco_results)
        elif stage == "test":
            self.test_coco_results.extend(coco_results)

        return coco_results
    
    def on_train_epoch_end(self):
        self.evaluate_epoch_end("train", self.train_coco_results, self.train_coco_results_json_fname)

    def on_validation_epoch_end(self):
        self.evaluate_epoch_end("val", self.val_coco_results, self.val_coco_results_json_fname)
    
    def on_test_epoch_end(self):
        self.evaluate_epoch_end("test", self.test_coco_results, self.test_coco_results_json_fname)

    def evaluate_epoch_end(self, stage, results, json_fname):

        with open(self.val_coco_results_json_fname, 'w') as f:
            json.dump(self.val_coco_results, f)

        # Load COCO annotations and results
        if stage == "train":
            coco_gt = self.trainer.datamodule.train_dataset.coco
        elif stage == "val":
            coco_gt = self.trainer.datamodule.val_dataset.coco
        else:
            coco_gt = self.trainer.datamodule.test_dataset.coco

        coco_dt = coco_gt.loadRes(self.val_coco_results_json_fname)

        delete_file(json_fname)

        # Initialize COCOeval object
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Log results with MLFlow
        self.log(f"{stage}_mAP", coco_eval.stats[0], sync_dist=True)
        self.log(f"{stage}_mAP_50", coco_eval.stats[1], sync_dist=True)
        self.log(f"{stage}_mAP_75", coco_eval.stats[2], sync_dist=True)
        self.log(f"{stage}_mAP_small", coco_eval.stats[3], sync_dist=True)
        self.log(f"{stage}_mAP_medium", coco_eval.stats[4], sync_dist=True)
        self.log(f"{stage}_mAP_large", coco_eval.stats[5], sync_dist=True)
        self.log(f"{stage}_AR_1", coco_eval.stats[6], sync_dist=True)
        self.log(f"{stage}_AR_10", coco_eval.stats[7], sync_dist=True)
        self.log(f"{stage}_AR_100", coco_eval.stats[8], sync_dist=True)
        self.log(f"{stage}_AR_small", coco_eval.stats[9], sync_dist=True)
        self.log(f"{stage}_AR_medium", coco_eval.stats[10], sync_dist=True)
        self.log(f"{stage}_AR_large", coco_eval.stats[11], sync_dist=True)

    def configure_optimizers(self):
        return None


def main():

    for repeat_id in [0,1,2,3,4]:

        data_dir = f'/mnt/data_vilen/05_model_input/repeated_nonstratified_train_val_test_split/yolo_v6_all_classes_COCO_v4_noEmpty/repeated_split_{repeat_id}'
        num_classes = 9  # Number of classes
        model_checkpoint = f'/mnt/data/06_models/resnet101_FINALv4_ResNet101_WeightsIMAGENET1K_V1_allClasses_v1_10ep__no_augm_nonStratSplit_repeat{repeat_id}/checkpoints/chkpnt_FINAL.ckpt'

        # Load the trained model
        model = FasterRCNNModule.load_from_checkpoint(model_checkpoint, num_classes=num_classes)
        model.eval()  # Set the model to evaluation mode
        model.freeze()

        def print_model_size_in_millions(model):
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model size: {total_params / 1_000_000:.2f}M parameters")

        print_model_size_in_millions(model)

        # Prepare the validation dataset and dataloader
        coco_data = CocoDataModule(data_dir, batch_size=4, take_subset=0)
        coco_data.setup()  # Explicitly call setup if necessary

        eval_module = EvaluationLightningModule(model)

        mlflow_logger = MLFlowLogger(
            experiment_name='FasterRCNN_Validation_v3_nonStrat_allClasses_v4',
            tracking_uri='http://mlflow-server:5000'
        )
        mlflow_logger.log_hyperparams({"repeat_id": repeat_id, 
                                       "stratified": False})

        trainer = Trainer(
            strategy=DDPStrategy(find_unused_parameters=False),
            accelerator='gpu',
            devices=-1,
            max_epochs=1,
            logger=mlflow_logger
        )

        # Evaluate training performance separately
        train_loader = coco_data.train_dataloader()
        for batch_idx, batch in enumerate(train_loader):
            eval_module.training_step(batch, batch_idx)
        eval_module.on_train_epoch_end()

        # Run validation
        trainer.validate(eval_module, datamodule=coco_data)

        # Run test evaluation
        trainer.test(eval_module, datamodule=coco_data)

if __name__ == "__main__":
    main()
