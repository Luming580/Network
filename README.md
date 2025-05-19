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
 --Process_dataset
  --classifier_dataset
  	---Annotation
    ------the label file
    ---ImageSets
    ------Main
    ----------train.txt
    ----------test.txt
    ----------val.txt
    ---JPEGImages
    ------the image files
  # 1.After completing the placement of the dataset, we need to use process.py to obtain 2025_train.txt and 2025_val.txt for training.
  # 2.Modify the parameters in process.py. During the first training, only the classes_path can be modified. The classes_path is used to 			point to the txt corresponding to the classifier category.
  # 3.When training all dataset, you can create a cls_classes.txt by yourself, and write the categories you need to distinguish in it. 			The content of the model_data/cls_classes.txt file is:
  DT_GAP
  DT_LACUNAS
  DT_CRACK
  PIP
  PIP_S
  GROUND_LAMP
  # 4.Modify the classes_path in process.py to make it correspond to cls_classes.txt, and run process.py
  python process.py
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

