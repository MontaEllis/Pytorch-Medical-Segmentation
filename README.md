# Pytorch Medical Segmentation
<i>Read Chinese Introduction：<a href='https://github.com/MontaEllis/Pytorch-Medical-Segmentation/blob/master/README-zh.md'>Here！</a></i><br />

## Recent Updates
* 2021.1.8 The train and test codes are released.


## Requirements
* pytorch1.7
* python>=3.6

## Notice
* You can modify **hparam.py** to determine whether 2D or 3D segmentation and whether multicategorization is possible.
* We provide backbone networks for almost all 2D and 3D segmentation.
* This repository is compatible with all medical data formats, by modifying **fold_arch** in **hparam.py** of the config.



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

## Examples
![](https://ellis.oss-cn-beijing.aliyuncs.com/img/20210108181532.png)



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

## By The Way
This project is not perfect and there are still many problems. If you are using this project and would like to give the author some feedback, you can send [Kangneng Zhou](elliszkn@163.com) an email or contact him to join a wechat group via scan:
![](https://ellis.oss-cn-beijing.aliyuncs.com/img/20210108181721.png)

## Acknowledgements
This repository is an unoffical PyTorch implementation of Medical segmentation in 3D and 2D and highly based on [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch) and [torchio](https://github.com/fepegar/torchio).Thank you for the above repo. Thank you to [Cheng Chen](b20170310@xs.ustb.edu.cn), [Daiheng Gao](samuel.gao023@gmail.com), [Jie Zhang](jpeter.zhang@connect.polyu.hk) and [Xing Tao](kakatao@foxmail.com) for all the help I received.