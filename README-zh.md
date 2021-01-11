# Pytorch Medical Segmentation
<i>英文版请戳：<a href='https://github.com/MontaEllis/Pytorch-Medical-Segmentation/blob/master/README.md'>这里！</a></i><br />


## 最近的更新
* 2021.1.8 训练和测试代码已经发布

## 环境要求
* pytorch1.7
* python>=3.6

## 通知
* 您可以修改**hparam.py**文件来确定是2D分割还是3D分割以及是否可以进行多分类。

* 我们几乎提供了所有的2D和3D分割的算法。
* 本项目兼容几乎所有的医学数据格式(例如 nii.gz, nii, mhd, nrrd, ...)，修改**hparam.py**的**fold_arch**即可。


## 训练
* 不使用预训练模型
```
python train.py
```
* 使用预训练模型
```
python train.py -k True
```
  
## Inference
* 测试
```
python test.py
```

## 实例
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

## By The Way
这个项目并不完美，还存在很多问题。如果您正在使用这个项目，并想给作者一些反馈，您可以给[Kangneng Zhou](elliszkn@163.com)发邮件.

## 致谢
这个项目是一个非官方PyTorch实现的3D和2D医学分割，高度依赖于[MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)和[torchio](https://github.com/fepegar/torchio)。感谢上述项目。感谢[Cheng Chen](b20170310@xs.ustb.edu.cn), [Daiheng Gao](https://github.com/tomguluson92), [Jie Zhang](jpeter.zhang@connect.polyu.hk)和[Xing Tao](kakatao@foxmail.com)对我的帮助。
