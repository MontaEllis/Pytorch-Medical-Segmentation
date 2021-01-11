# Pytorch Medical Segmentation
<i>Read Chinese Introduction：<a href='https://github.com/MontaEllis/Pytorch-Medical-Segmentation/blob/master/README-zh.md'>Here！</a></i><br />

## Recent Updates
* 2021.1.8 The train and test codes are released.


## Requirements
* pytorch1.7
* python>=3.6

## Notice
* You can modify **hparam.py** to determine whether 2D or 3D segmentation and whether multicategorization is possible.
* We provide algorithms for almost all 2D and 3D segmentation.
* This repository is compatible with almost all medical data formats(e.g. nii.gz, nii, mhd, nrrd, ...), by modifying **fold_arch** in **hparam.py** of the config.



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
![](https://ellis.oss-cn-beijing.aliyuncs.com/img/20210108185333.png)



## Done
* 2D
- [x] unet
- [x] unet++
- [x] miniseg
- [x] segnet
- [x] pspnet
- [x] highresnet(copy from https://github.com/fepegar/highresnet, Thank you to [fepegar](https://github.com/fepegar) for your generosity!)
- [x] deeplab
- [x] fcn
* 3D
- [x] unet3d
- [x] densevoxelnet3d
- [x] fcn3d
- [x] vnet3d
- [x] highresnert(copy from https://github.com/fepegar/highresnet, Thank you to [fepegar](https://github.com/fepegar) for your generosity!)
- [x] densenet3d

## TODO
- [ ] dataset
- [ ] benchmark
- [ ] nnunet

## By The Way
This project is not perfect and there are still many problems. If you are using this project and would like to give the author some feedbacks, you can send [Kangneng Zhou](elliszkn@163.com) an email.

## Acknowledgements
This repository is an unoffical PyTorch implementation of Medical segmentation in 3D and 2D and highly based on [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch) and [torchio](https://github.com/fepegar/torchio).Thank you for the above repo. Thank you to [Cheng Chen](b20170310@xs.ustb.edu.cn), [Daiheng Gao](https://github.com/tomguluson92), [Jie Zhang](jpeter.zhang@connect.polyu.hk) and [Xing Tao](kakatao@foxmail.com) for all the help I received.
