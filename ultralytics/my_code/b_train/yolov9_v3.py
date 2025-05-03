from ultralytics import YOLO



for split_no in [0,1,2,3,4]:
    model = YOLO("yolov9e.pt")

    # Load YOLOv10n model from scratch
    # model = YOLO("yolov10n.yaml")

    # Train the model
    model.train(data=f"/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/ultralitics_all_classes_COCO_v4_noEmpty/repeated_split_{split_no}/data.yml", 
                epochs=100, 
                imgsz=640,
                batch=16,
                device="0,1")