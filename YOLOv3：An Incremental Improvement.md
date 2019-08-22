# YOLOv3: An Incremental Improvement

论文地址：<https://arxiv.org/abs/1804.02767>

代码复现：<https://github.com/qqwweee/keras-yolo3> （Keras实现）

&emsp;&emsp;&emsp;&emsp;&emsp;<https://github.com/eriklindernoren/PyTorch-YOLOv3>（Pytorch实现）

&emsp;&emsp;&emsp;&emsp;&emsp;<https://github.com/YunYang1994/tensorflow-yolov3>  （TensorFlow实现）



## 一、介绍

首先作者把RetinaNet论文中的实验结果图拿出来，说明YOLOv3尽管mAP不如两阶段检测器，但是速度还是要快很多。可以看到，YOLOv3图片Input_size从320到416再到608，速度越慢但准确性越好。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv3/1.png)



## 二、主要内容

#### 相对于Yolov2的改变
（1）使用残差模型；YOLOv3的特征提取器是一个残差模型，因为包含53个卷积层，所以称为Darknet-53，从网络结构上看，相比Darknet-19网络使用了残差单元，所以可以构建得更深。
（2）采用FPN架构，具体来说是三个尺度的特征图，例如416x416的输入图片，使用13x13,26x26,52x52的特征图；


#### bbox预测

与YOLOv2一样，使用anchor来预测bbox，每个bbox包括4个坐标值tx, ty, tw, th；如果cell相对图片的左上角偏离(cx, cy)，并且bbox的宽高为pw和ph，则预测坐标为：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv3/2.png)

训练阶段使用误差平方和损失。

YOLOv3使用logistic regression为每个bbox预测含有物体的score。如果某个bbox和gt的IoU最大，则score为1；使用nms去掉和gt IoU大于threshold（一般取0.5）的bbox。和Faster R-CNN不同，这里只让一个bbox负责某个gt框，其他的框不会有坐标或分类损失，只有是否含有物体的损失；



#### 分类预测

作者认为softmax不是必须的，因此这里使用了独立的逻辑回归分类器；在训练过程中使用二分类交叉熵损失。这种做法对不同label可能重叠（如woman和person）的数据集很有帮助。

**Anchor先验参数计算**
这里计算训练数据的anchor是根据模型中图像的输入尺寸得到的，即无需转化可以直接拿来训练（其实就是乘了一下输入尺寸）。参考：https://github.com/lars76/kmeans-anchor-boxes。
```
# -*- coding=utf-8 -*-
import glob
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from kmeans import kmeans, avg_iou

# 根文件夹
ROOT_PATH = '/data/DataBase/YOLO_Data/V3_DATA/'
# 聚类的数目
CLUSTERS = 6
# 模型中图像的输入尺寸，默认是一样的
SIZE = 640

# 加载YOLO格式的标注数据
def load_dataset(path):
    jpegimages = os.path.join(path, 'JPEGImages')
    if not os.path.exists(jpegimages):
        print('no JPEGImages folders, program abort')
        sys.exit(0)
    labels_txt = os.path.join(path, 'labels')
    if not os.path.exists(labels_txt):
        print('no labels folders, program abort')
        sys.exit(0)

    label_file = os.listdir(labels_txt)
    print('label count: {}'.format(len(label_file)))
    dataset = []

    for label in label_file:
        with open(os.path.join(labels_txt, label), 'r') as f:
            txt_content = f.readlines()

        for line in txt_content:
            line_split = line.split(' ')
            roi_with = float(line_split[len(line_split)-2])
            roi_height = float(line_split[len(line_split)-1])
            if roi_with == 0 or roi_height == 0:
                continue
            dataset.append([roi_with, roi_height])
            # print([roi_with, roi_height])

    return np.array(dataset)

data = load_dataset(ROOT_PATH)
out = kmeans(data, k=CLUSTERS)

print(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}-{}".format(out[:, 0] * SIZE, out[:, 1] * SIZE))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))
```
经过运行之后得到一组如下数据：
```
[[0.21203704 0.02708333]
 [0.34351852 0.09375   ]
 [0.35185185 0.06388889]
 [0.29513889 0.06597222]
 [0.24652778 0.06597222]
 [0.24861111 0.05347222]]

Accuracy: 89.58%
Boxes:
[135.7037037  219.85185185 225.18518519 188.88888889 157.77777778 159.11111111] - [17.33333333 60. 40.88888889 42.22222222 42.22222222 34.22222222]
```
其中的Boxes就是得到的anchor参数，以上面给出的计算结果为例，最后的anchor参数设置为
anchors = 135,17,  219,60,  225,40,  188,42,  157,42,  159,34 

为什么YOLOv2和YOLOv3的anchor大小有明显区别？
**在YOLOv2中，作者用最后一层feature map的相对大小来定义anchor大小**。也就是说，在YOLOv2中，最后一层feature map大小为**13X13**（不同输入尺寸的图像最后的feature map也不一样的），相对的anchor大小范围就在（0x0，13x13]，**如果一个anchor大小是9x9，那么其在原图上的实际大小是288x288**。

**而在YOLOv3中，作者又改用相对于原图的大小来定义anchor**，anchor的大小为（0x0，input_w x input_h]。所以，在两份cfg文件中，anchor的大小有明显的区别。如下是作者自己的解释：
So YOLOv2 I made some design choice errors, I made the anchor box size be relative to the feature size in the last layer. Since the network was down-sampling by 32. This means it was relative to 32 pixels so an anchor of 9x9 was actually 288px x 288px.
In YOLOv3 anchor sizes are actual pixel values. this simplifies a lot of stuff and was only a little bit harder to implement
https://github.com/pjreddie/darknet/issues/555#issuecomment-376190325

Yolov3中，在416x416尺度下，最小可以感受 8x8 像素的的信息，即针对52x52的feature map，如下图
![这里随便写文字](https://pic1.zhimg.com/80/v2-21ab25791e7437631e5cba5aec36691c_hd.jpg)

#### 多尺度预测

YOLOv3借鉴了FPN的思想，使用三种不同尺度的特征图来进行预测；预测结果是一个3维的tensor，每个维度分别描述了bbox坐标，是否含有物体（score）以及分类；比如对于COCO数据集，每个尺度预测3个框，则tensor的数量为 N x N x [3 x (4 + 1 + 80)]，其中包含了4个bbox坐标偏移，1个score以及80个类别预测。

**使用2 layers previous 的feature map，做2倍上采样，并进行融合，这样能够获得上层更丰富的语义信息**。后面在跟几个卷积层，来更好地组合之前融合的feature map的特征，最终预测一个两倍大小的tensor。
![这里随便写文字](https://pic4.zhimg.com/80/v2-ffbc5b713c98c13e2659bb528b05fd67_hd.jpg)

同样使用K-means算法，作者在COCO数据集上得到的9个anchor：10x13, 16x30, 33x23, 30x61, 62x45, 59x119, 116x90, 156x198, 373x326；



#### 特征提取

这里使用了一种新的网络来提取特征。网络结合了YOLOv2中使用的网络，以及Darknet-19和新型的残差网络块；使用连续的3x3和1x1卷积，加入shortcut连接。总共有53个卷积层，因此称为Darknet-53，结构如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv3/3.png)

这个网络比Darknet-19性能强很多，和ResNet-101和ResNet-152相比速度又很快。



#### 训练

使用全图进行训练，没有减少负样本的数量（with no hard negative mining）。使用了多尺度训练、数据增强、BN。



## 三、实验结果

结果如下图所示，可以看出，**Yolov2、SSD都不适合小目标检测，RetinaNet最适合但耗时较长**，YOLOv3准确性虽然不及RetinaNet，但是用时少很多。另外YOLOv3在“IoU=0.5”这种古老的评价标准中效果拔群，mAP达到了57.9，略低于Faster R-CNN + ResNet-101-FPN的59.1，以及RetinaNet + ResNet-101-FPN也是59.1；但是当IoU进一步提升到0.75时，mAP显著降低，说明YOLOv3很难保证预测的bbox和实际物体有较好的贴合（get the boxes perfectly aligned with the box）。另外，YOLO之前在小物体领域一直表现不佳，但YOLOv3通过多尺度预测，可以很好地检测到小物体，效果和Faster R-CNN + ResNet-101-FPN持平，略逊色于RetinaNet + ResNet-101-FPN，但是在较大物体方面又表现不佳了，所以作者讲需要做的工作还很多。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv3/4.png)



此外，作者发现有一些操作会降低mAP，包括：1、使用传统方法预测anchor box的x，y偏移  2、使用linear方法预测x，y而不用logistic  3、加入Focal loss  4、Dual IoU thresholds and truth assignment；



## 四、结论

* 
* 
