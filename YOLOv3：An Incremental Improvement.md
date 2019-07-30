# **YOLOv3: An Incremental Improvement **

论文地址：<https://arxiv.org/abs/1804.02767>

代码复现：<https://github.com/qqwweee/keras-yolo3> （Keras实现）

&emsp;&emsp;&emsp;&emsp;&emsp;<https://github.com/eriklindernoren/PyTorch-YOLOv3>（Pytorch实现）

&emsp;&emsp;&emsp;&emsp;&emsp;<https://github.com/YunYang1994/tensorflow-yolov3>  （TensorFlow实现）



## 一、介绍

首先作者把RetinaNet论文中的实验结果图拿出来，说明YOLOv3尽管mAP不如两阶段检测器，但是速度还是要快很多，见下图。可以看到，YOLOv3图片Input_size从320到416再到608，速度越慢但准确性越好。和YOLOv2相比，YOLOv3做到了更快，更好；

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv3/1.png)



## 二、blob主要内容

#### bbox预测

与YOLOv2一样，使用anchor来预测bbox，每个bbox包括4个坐标值tx, ty, tw, th；如果cell相对图片的左上角偏离(cx, cy)，并且bbox的宽高为pw和ph，则预测坐标为：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv3/2.png)

训练阶段使用误差平方和损失。

YOLOv3使用logistic regression为每个bbox预测含有物体的score。如果某个bbox和gt的IoU最大，则score为1；使用nms去掉和gt IoU大于threshold（一般取0.5）的bbox。和Faster R-CNN不同，这里只让一个bbox负责某个gt框，其他的框不会有坐标或分类损失，只有是否含有物体的损失；



#### 分类预测

作者认为softmax不是必须的，因此这里使用了独立的逻辑回归分类器；在训练过程中使用二分类交叉熵损失。这种做法对不同label可能重叠（如woman和person）的数据集很有帮助。



#### 多尺度预测

YOLOv3借鉴了FPN的思想，使用三种不同尺度的特征图来进行预测；预测结果是一个3维的tensor，每个维度分别描述了bbox坐标，是否含有物体（score）以及分类；比如对于COCO数据集，每个尺度预测3个框，则tensor的数量为 N x N x [3 x (4 + 1 + 80)]，其中包含了4个bbox坐标偏移，1个score以及80个类别预测。

使用2 layers previous（？？）的feature map，做2倍上采样，并进行融合，这样能够获得上层更丰富的语义信息。后面在跟几个卷积层，来更好地组合之前融合的feature map的特征，最终预测一个两倍大小的tensor。？？

同样使用K-means算法，作者在COCO数据集上得到的9个anchor：10x13, 16x30, 33x23, 30x61, 62x45, 59x119, 116x90, 156x198, 373x326；



#### 特征提取

这里使用了一种新的网络来提取特征。网络结合了YOLOv2中使用的网络，以及Darknet-19和新型的残差网络块；使用连续的3x3和1x1卷积，加入shortcut连接。总共有53个卷积层，因此称为Darknet-53，结构如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv3/3.png)

这个网络比Darknet-19性能强很多，和ResNet-101和ResNet-152相比速度又很快。



#### 训练

使用全图进行训练，没有减少负样本的数量（with no hard negative mining）。使用了多尺度训练、数据增强、BN。



## 三、实验结果

结果如下图所示，可以看出，Yolov2、SSD都不适合小目标检测，RetinaNet最适合但耗时较长，YOLOv3准确性虽然不及RetinaNet，但是用时少很多。另外YOLOv3在“IoU=0.5”这种古老的评价标准中效果拔群，mAP达到了57.9，略低于Faster R-CNN + ResNet-101-FPN的59.1，以及RetinaNet + ResNet-101-FPN也是59.1；但是当IoU进一步提升到0.75时，mAP显著降低，说明YOLOv3很难保证预测的bbox和实际物体有较好的贴合（get the boxes perfectly aligned with the box）。另外，YOLO之前在小物体领域一直表现不佳，但YOLOv3通过多尺度预测，可以很好地检测到小物体，效果和Faster R-CNN + ResNet-101-FPN持平，略逊色于RetinaNet + ResNet-101-FPN，但是在较大物体方面又表现不佳了，所以作者讲需要做的工作还很多。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv3/4.png)



此外，作者发现有一些操作会降低mAP，包括：1、使用传统方法预测anchor box的x，y偏移  2、使用linear方法预测x，y而不用logistic  3、加入Focal loss  4、Dual IoU thresholds and truth assignment；



## 四、结论

* 
* 
