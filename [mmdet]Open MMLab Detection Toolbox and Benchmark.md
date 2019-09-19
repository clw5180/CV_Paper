# **MMDetection: Open MMLab Detection Toolbox and Benchmark **

论文地址：https://arxiv.org/abs/1906.07155

项目地址：https://github.com/open-mmlab/mmdetection



## 一、介绍

mmdetection框架（toolbox）主要用于目标检测和实例分割。相比于其他框架的特色在于**模块化的设计（如backbone-neck-head）**，可以通过不同部分的组合来实现特定的网络。另外就是提供了很多通用的模块和好用的工具，比如Soft NMS、DCNv2等，以及数据增强。



## 二、主要内容

#### 支持的模型

- **一阶段**：包括2015年提出的SSD、2017年提供的RetinaNet和2019年提出的GHM（a gradient harmonizing mechanism）、FCOS（a fully convolutional anchor-free single stage detector）、FSAF（a feature selective anchor-free module ）。
- **两阶段**：包括2015年提出的Fast R-CNN和Faster R-CNN，2016年提出的R-FCN（全卷积网络，速度快于Faster）、2017年提出的Mask R-CNN、2018年提出的Grid R-CNN和2019年提出的Mask Scoring R-CNN和Double-Head R-CNN。
- **多阶段**：包括2017年提出的Cascade R-CNN和2019年提出的Hybrid Task Cascade（a multi-stage multi-branch object detection and instance segmentation method）。



#### 通用的模块和方法

• **Mixed Precision Training** [22]: train deep neural networks using half precision floating point (FP16) numbers, proposed in 2018.
• **Soft NMS** [1]: an alternative to NMS, proposed in 2017.
• **OHEM** [29]: an online sampling method that mines hard samples for training, proposed in 2016.
• DCN [8]: deformable convolution and deformable RoI
pooling, proposed in 2017.
• **DCNv2** [42]: modulated deformable operators, proposed in 2018.
• **Train from Scratch** [12]: training from random initialization instead of ImageNet pretraining, proposed in
2018.

• ScratchDet [40]: another exploration on training from scratch, proposed in 2018.
• **M2Det** [38]: a new feature pyramid network to construct more effective feature pyramids, proposed in
2018.

• **GCNet** [3]: global context block that can efficiently model the global context, proposed in 2019.
• **Generalized Attention** [41]: a generalized attention formulation, proposed in 2019.
• **SyncBN** [25]: synchronized batch normalization across GPUs, we adopt the official implementation by
PyTorch.
• **Group Normalization** [36]: a simple alternative to BN, proposed in 2018.
• Weight Standardization [26]: standardizing the weights in the convolutional layers for micro-batch
training, proposed in 2019.
• **HRNet** [30, 31]: a new backbone with a focus on learning reliable high-resolution representations, proposed in 2019.
• **Guided Anchoring** [34]: a new anchoring scheme that predicts sparse and arbitrary-shaped anchors, proposed in 2019.
• Libra R-CNN [23]: a new framework towards balanced learning for object detection, proposed in 2019. 

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/R2CNN/1.png)





## 三、实验结果





## 四、结论

* 
* 

&emsp;