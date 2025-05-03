import os
import sys
sys.path.append('/mnt/female-data-eval')
sys.path.append('/mnt/female-data-eval/d_training')
import tempfile
from tqdm.auto import tqdm
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from pycocotools.cocoeval import COCOeval
import json

from pathlib import Path
from d_training.faster_rcnn_resnet50_v2_nonstrat import CocoDataModule, FasterRCNNModule
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
                 model):
        
        super().__init__()
        self.model = model
        self.train_coco_results = []
        self.val_coco_results = []
        self.test_coco_results = []

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
        self.evaluate_epoch_end("train", self.train_coco_results)

    def on_validation_epoch_end(self):
        self.evaluate_epoch_end("val", self.val_coco_results)
    
    def on_test_epoch_end(self):
        self.evaluate_epoch_end("test", self.test_coco_results)

    def evaluate_epoch_end(self, stage, results):
        
        # Load COCO annotations and results
        if stage == "train":
            coco_gt = self.trainer.datamodule.train_dataset.coco
        elif stage == "val":
            coco_gt = self.trainer.datamodule.val_dataset.coco
        else:
            coco_gt = self.trainer.datamodule.test_dataset.coco

        with tempfile.NamedTemporaryFile(delete=True, suffix='.json') as temp:
            with open(temp.name, 'w') as temp_file:
                json.dump(results, temp_file)

            try:
                coco_dt = coco_gt.loadRes(temp.name)
            except Exception as e:
                print(e)
                zero_stats = [torch.tensor(0.0, device=self.device)] * 12  # Ensure zero_stats is on the correct device
                self.log_metrics(stage, zero_stats)
                return
            
        # Initialize COCOeval object
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Convert stats to tensors and move to the correct device for synchronization
        stats = [torch.tensor(stat, device=self.device) for stat in coco_eval.stats]

        # Log results with MLFlow
        self.log_metrics(stage, stats)

    def log_metrics(self, stage, stats):
        self.log(f"{stage}_mAP", stats[0], sync_dist=True, rank_zero_only=True)
        self.log(f"{stage}_mAP_50", stats[1], sync_dist=True, rank_zero_only=True)
        self.log(f"{stage}_mAP_75", stats[2], sync_dist=True, rank_zero_only=True)
        self.log(f"{stage}_mAP_small", stats[3], sync_dist=True, rank_zero_only=True)
        self.log(f"{stage}_mAP_medium", stats[4], sync_dist=True, rank_zero_only=True)
        self.log(f"{stage}_mAP_large", stats[5], sync_dist=True, rank_zero_only=True)
        self.log(f"{stage}_AR_1", stats[6], sync_dist=True, rank_zero_only=True)
        self.log(f"{stage}_AR_10", stats[7], sync_dist=True, rank_zero_only=True)
        self.log(f"{stage}_AR_100", stats[8], sync_dist=True, rank_zero_only=True)
        self.log(f"{stage}_AR_small", stats[9], sync_dist=True, rank_zero_only=True)
        self.log(f"{stage}_AR_medium", stats[10], sync_dist=True, rank_zero_only=True)
        self.log(f"{stage}_AR_large", stats[11], sync_dist=True, rank_zero_only=True)

    def configure_optimizers(self):
        return None

    def evaluate_training(self, trainer):
        # Run training evaluation using trainer
        trainer.validate(self, dataloaders=[self.trainer.datamodule.train_dataloader()])


def main():

    models_final_ckpt = {
        0: "/mnt/48/d17825c40cd544ebb66cb3de0a25f49e/checkpoints/epoch=9-step=15950.ckpt",
        1: "/mnt/48/a214481c4d5e47f992c76b6c4e8623ea/checkpoints/epoch=9-step=15970.ckpt",
        2: "/mnt/48/b3f1fde2368e45d2a48b524415f81950/checkpoints/epoch=9-step=16070.ckpt",
        3: "/mnt/48/7e5231e8504b4f95bafd56355770f083/checkpoints/epoch=9-step=14530.ckpt",
        4: "/mnt/48/a185b1d72d6449f7b26e81e5d7f6974a/checkpoints/epoch=9-step=15470.ckpt"
    }

    for repeat_id in [0, 1, 2, 3, 4]:
        print('=' * 60)
        print('=' * 60)
        print('=' * 60)
        print(f'REPEAT {repeat_id}')
        print('-' * 30)
        data_dir = f'/mnt/data_vilen/05_model_input/repeated_nonstratified_train_val_test_split/yolo_v6_all_classes_COCO_v4_noEmpty/repeated_split_{repeat_id}'
        num_classes = 9  # Number of classes
        model_checkpoint = models_final_ckpt[repeat_id]  #f'/mnt/data/06_models/resnet101_FINALv4_ResNet101_WeightsIMAGENET1K_V1_allClasses_v1_10ep__no_augm_nonStratSplit_repeat{repeat_id}/checkpoints/chkpnt_FINAL.ckpt'

        # Load the trained model
        model = FasterRCNNModule.load_from_checkpoint(model_checkpoint, num_classes=num_classes)
        model.eval()  # Set the model to evaluation mode
        model.freeze()

        def print_model_size_in_millions(model):
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model size: {total_params / 1_000_000:.2f}M parameters")

        print_model_size_in_millions(model)

        # Prepare the datasets and dataloaders
        coco_data = CocoDataModule(data_dir, batch_size=8, take_subset=0)
        coco_data.setup()  # Explicitly call setup if necessary

        eval_module = EvaluationLightningModule(model)

        mlflow_logger = MLFlowLogger(
            experiment_name='FasterRCNN_Evaluation_v3_nonStrat_allClasses_v20_23Jun24',
            tracking_uri='http://mlflow-server:5000'
        )
        mlflow_logger.log_hyperparams({"repeat_id": repeat_id, 
                                       "stratified": False})

        trainer = Trainer(
            strategy=DDPStrategy(find_unused_parameters=False),
            accelerator='gpu',
            devices=-1,
            # devices=[0],
            max_epochs=1,
            logger=mlflow_logger
        )

        # Run validation
        trainer.validate(eval_module, datamodule=coco_data)

        # Run test evaluation
        trainer.test(eval_module, datamodule=coco_data)

        # Evaluate training performance separately
        eval_module.evaluate_training(trainer)


if __name__ == "__main__":
    main()
