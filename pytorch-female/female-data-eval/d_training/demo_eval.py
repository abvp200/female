from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annType = ['segm','bbox','keypoints']
annType = annType[1]

cocoGt=COCO('/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/yolo_v6_all_classes_COCO/repeated_split_0/annotations/instances_val.json')

