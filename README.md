## Adaptive Combination Window Spatial-Frequency Transformation Network  for Ground Penetrating Radar B-Scan Images Reconstruction

---

## Dataset

```python
# Process the Adaptive Combination Window Spatial-Frequency Transformation Network dataset
--dataset
  --train
  	---original
    ------.png
    ---target
    ------.png
  --test
    ---original
    ------.png
    ---target
    ------.png
  --val
    ---original
    ------.png
    ---target
    ------.png
  # And process the defect classification guided dataset
  # 1.Before training, place the label file in the Annotation under the VOC2007 folder in the VOCdevkit folder.
  # 2.Before training, place the image files in JPEGImages under the VOC2007 folder in the VOCdevkit folder.
  # 3.After completing the placement of the dataset, we need to use voc_annotation.py to obtain 2007_train.txt and 2007_val.txt for training.
  # 4.Modify the parameters in voc_annotation.py. During the first training, only the classes_path can be modified. The classes_path is used to point to the txt corresponding to the detection category.
  # 5.When training your own dataset, you can create a cls_classes.txt by yourself, and write the categories you need to distinguish in it. The content of the model_data/cls_classes.txt file is:
  DT_GAP
  DT_LACUNAS
  DT_CRACK
  PIP
  PIP_S
  GROUND_LAMP
  # 6.Modify the classes_path in voc_annotation.py to make it correspond to cls_classes.txt, and run voc_annotation.py
```



## Train

```python
  python Tranchange.py # Modify the allocation of the dataset corresponding to the option.py data path
```



## Predict

```python
  cd Run
  python autotest.py # Modify the weight file
```



## 所需环境

```C++
- PyTorch == 1.13.0
- TorchVision == 0.14.0
- CUDA 11.8 
```

