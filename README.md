## Recent Updates
* 2021.1.8 The train and test codes are released.


## Requirements
* pytorch1.7
* python>=3.6

## Notice
* You can modify **config.py** to determine whether 2D or 3D segmentation and whether multicategorization is possible.
* We provide backbone networks for almost all 2D and 3D segmentation.



## Training
* without pretrained-model
```
python train.py
```
* with pretrained-model
```
python train.py -k True
```
  
## Inference
* testing
```
python test.py
```


## TODO
* 2D
    - [x] unet
    - [x] unet++
    - [x] miniseg
    - [x] segnet
    - [x] pspnet
    - [x] highresnet
    - [x] deeplab
    - [x] fcn
* 3D
    - [x] unet3d
    - [x] densevoxelnet3d
    - [x] fcn3d
    - [x] vnet3d
    - [x] highresnert
    - [x] densenet3d


## Acknowledgements
This repository is an unoffical PyTorch implementation of Medical segmentation in 3D and 2D and highly based on [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch) and [torchio](https://github.com/fepegar/torchio).Thank you for the above repo. Thank you to [Cheng Chen](b20170310@xs.ustb.edu.cn) for all the help I received.