from ultralytics import YOLO


def freeze(layer_name: str, l_from: int, l_to: int) -> bool:    
    l_name = layer_name.replace('model.model.', '')
    l_names = l_name.split('.')
    layer_num = int(l_names[0])
    return l_from <= layer_num and layer_num < l_to


def freeze_early(layer_name: str) -> bool:
    return freeze(layer_name, 1, 23)  # freeze all layers from 1 to 22 (not 23) 


def freeze_mid(layer_name: str) -> bool:
    return freeze(layer_name, 1, 33)  # freeze all layers from 1 to 32 (not 33)


def freeze_all(layer_name: str) -> bool:
    return freeze(layer_name, 1, 42)  # freeze all layers from 1 to 42 (not 43)


def test():
    assert True == freeze_early('model.model.2.bn.bias')
    assert True == freeze_mid('model.model.2.bn.bias')
    assert True == freeze_all('model.model.2.bn.bias')

    assert True == freeze_early('model.model.22.cv2.0.m.0.cv2.bn.weight')
    assert True == freeze_mid('model.model.22.cv2.0.m.0.cv2.bn.weight')
    assert True == freeze_all('model.model.22.cv2.0.m.0.cv2.bn.weight')

    assert False == freeze_early('model.model.28.cv2.0.m.1.cv2.bn.bias')
    assert True == freeze_mid('model.model.28.cv2.0.m.1.cv2.bn.bias')
    assert True == freeze_all('model.model.28.cv2.0.m.1.cv2.bn.bias')

    assert False == freeze_early('model.model.41.cv2.0.m.1.cv2.bn.weight')
    assert False == freeze_mid('model.model.41.cv2.0.m.1.cv2.bn.weight')
    assert True == freeze_all('model.model.41.cv2.0.m.1.cv2.bn.weight')

    assert False == freeze_early('model.model.42.cv3.2.1.bn.bias')
    assert False == freeze_mid('model.model.42.cv3.2.1.bn.bias')
    assert False == freeze_all('model.model.42.cv3.2.1.bn.bias')


test()


for freezing_strategy in [freeze_mid]:  #[freeze_early, freeze_mid, freeze_all]:
    print('='*80)
    for split_no in [0,1,2,3,4]:
        model = YOLO("yolov9e.pt")

        # Print the model architecture to identify the layers
        # print(model)

        # Freeze layers
        for name, param in model.named_parameters():
            if freezing_strategy(name):
                param.requires_grad = False

        # Print the layers to verify
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

        # Train the model
        model.train(data=f"/mnt/data_vilen/05_model_input/repeated_stratified_train_val_test_split/ultralitics_all_classes_COCO_v4_noEmpty/repeated_split_{split_no}/data.yml", 
                    epochs=20, 
                    imgsz=1024,
                    batch=4,
                    device="0,1")