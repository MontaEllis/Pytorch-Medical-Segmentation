# Pytorch Medical Segmentation
<i>Read Chinese Introduction：<a href='https://github.com/MontaEllis/Pytorch-Medical-Segmentation/blob/master/README-zh.md'>Here！</a></i><br />

## Recent Updates
* 2021.1.8 The train and test codes are released.
* 2021.2.6 A bug in dice was fixed with the help of [Shanshan Li](https://github.com/ssli23).
* 2021.2.24 A video tutorial was released(https://www.bilibili.com/video/BV1gp4y1H7kq/).
* 2021.5.16 A bug in Unet3D implement was fixed.
* 2021.5.16 The metric code is released.
* 2021.6.24 All parameters can be adjusted in hparam.py.
* 2021.7.7 Now you can refer medical classification in [Pytorch-Medical-Classification](https://github.com/MontaEllis/Pytorch-Medical-Classification)

## Requirements
* pytorch1.7
* torchio<=0.18.20
* python>=3.6

## Notice
* You can modify **hparam.py** to determine whether 2D or 3D segmentation and whether multicategorization is possible.
* We provide algorithms for almost all 2D and 3D segmentation.
* This repository is compatible with almost all medical data formats(e.g. nii.gz, nii, mhd, nrrd, ...), by modifying **fold_arch** in **hparam.py** of the config. **I would like you to convert both the source and label images to the same type before using them, where labels are marked with 1, not 255.**
* If you want to use a **multi-category** program, please modify the corresponding codes by yourself. I cannot identify your specific categories.
* Whether in 2D or 3D, this project is processed using **patch**. Therefore, images do not have to be strictly the same size. In 2D, however, you should set the patch large enough.

## Prepare Your Dataset
### Example1
if your source dataset is :
```
source_dataset
├── source_1.mhd
├── source_1.zraw
├── source_2.mhd
├── source_2.zraw
├── source_3.mhd
├── source_3.zraw
├── source_4.mhd
├── source_4.zraw
└── ...
```

and your label dataset is :
```
label_dataset
├── label_1.mhd
├── label_1.zraw
├── label_2.mhd
├── label_2.zraw
├── label_3.mhd
├── label_3.zraw
├── label_4.mhd
├── label_4.zraw
└── ...
```

then your should modify **fold_arch** as **\*.mhd**, **source_train_dir** as **source_dataset** and **label_train_dir** as **label_dataset** in **hparam.py**

### Example2
if your source dataset is :
```
source_dataset
├── 1
    ├── source_1.mhd
    ├── source_1.zraw
├── 2
    ├── source_2.mhd
    ├── source_2.zraw
├── 3
    ├── source_3.mhd
    ├── source_3.zraw
├── 4
    ├── source_4.mhd
    ├── source_4.zraw
└── ...
```

and your label dataset is :
```
label_dataset
├── 1
    ├── label_1.mhd
    ├── label_1.zraw
├── 2
    ├── label_2.mhd
    ├── label_2.zraw
├── 3
    ├── label_3.mhd
    ├── label_3.zraw
├── 4
    ├── label_4.mhd
    ├── label_4.zraw
└── ...
```

then your should modify **fold_arch** as **\*/\*.mhd**, **source_train_dir** as **source_dataset** and **label_train_dir** as **label_dataset** in **hparam.py**


## Training
* without pretrained-model
```
set hparam.train_or_test to 'train'
python main.py
```
* with pretrained-model
```
set hparam.train_or_test to 'train'
python main.py -k True
```
  
## Inference
* testing
```
set hparam.train_or_test to 'test'
python main.py
```

## Examples
![](https://ellis.oss-cn-beijing.aliyuncs.com/img/20210108185333.png)
![](https://ellis.oss-cn-beijing.aliyuncs.com/img/2021-02-06%2022-40-07%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

## Tutorials
* https://www.bilibili.com/video/BV1gp4y1H7kq/

## Done
### Network
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
- [x] residual-unet3d
- [x] densevoxelnet3d
- [x] fcn3d
- [x] vnet3d
- [x] highresnert(copy from https://github.com/fepegar/highresnet, Thank you to [fepegar](https://github.com/fepegar) for your generosity!)
- [x] densenet3d

### Metric
- [x] metrics.py to evaluate your results

## TODO
- [ ] dataset
- [ ] benchmark
- [ ] nnunet

## By The Way
This project is not perfect and there are still many problems. If you are using this project and would like to give the author some feedbacks, you can send [Kangneng Zhou](elliszkn@163.com) an email, his **wechat** number is: ellisgege666

## Acknowledgements
This repository is an unoffical PyTorch implementation of Medical segmentation in 3D and 2D and highly based on [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch) and [torchio](https://github.com/fepegar/torchio).Thank you for the above repo. Thank you to [Cheng Chen](b20170310@xs.ustb.edu.cn), [Daiheng Gao](https://github.com/tomguluson92), [Jie Zhang](jpeter.zhang@connect.polyu.hk), [Xing Tao](kakatao@foxmail.com), [Weili Jiang](1379252229@qq.com) and [Shanshan Li](https://github.com/ssli23) for all the help I received.
