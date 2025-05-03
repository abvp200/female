from ultralytics import YOLO

import sys

sys.path.append("/mnt/my_code")
# sys.path.append("/mnt/my_code/d_training")

from c_eval.eval_util import class_map

validation = True

for split_no in [0,1,2,3,4]:
    model = YOLO("/mnt/runs/detect/train104/weights/best.pt")

    # Load YOLOv10n model from scratch
    # model = YOLO("yolov10n.yaml")

    # Train the model
    model.train(data=f"/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/ultralitics_all_classes_COCO_v4_noEmpty/repeated_split_{split_no}/data.yml", 
                epochs=100, 
                imgsz=640,
                batch=16,
                device="0,1")
    
    # Prepare the datasets and dataloaders
    coco_data = CocoDataModule(data_dir, batch_size=8, take_subset=0)
    coco_data.setup()  # Explicitly call setup if necessary

    if validation:
        val_dataloader = coco_data.val_dataloader()
    else:
        val_dataloader = coco_data.test_dataloader()
        
    detection_validator = DetectionValidator(dataloader=val_dataloader, device=torch.device('cuda:0'))
    detection_validator.init_metrics(model, class_map)
    
    
    model.val(data=f"/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/ultralitics_all_classes_COCO_v4_noEmpty/repeated_split_{split_no}/data.yml", 
                # epochs=100, 
                imgsz=640,
                batch=16,
                device="1",
                validator=detection_validator)