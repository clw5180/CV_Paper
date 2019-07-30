# **YOLOv1：You Only Look Once Unified, Real Time Object Detection**

论文地址：<http://arxiv.org/abs/1506.02640>

代码复现：<https://github.com/abhi2610/ohem>

&emsp;&emsp;&emsp;&emsp;&emsp;https://github.com/pjreddie/darknet

&emsp;&emsp;&emsp;&emsp;&emsp;https://github.com/hizhangp/yolo_tensorflow （tensorflow版本）



## 一. 介绍

&emsp;&emsp;一阶段方法（End to End方法）主要有SSD系列，YOLO系列，这种方法是将目标边框的定位问题转化为回归问题处理。由于思想的不同，二阶段检测方法在检测准确率和定位精度有优势，一阶段检测方法在速度上占有优势。

&emsp;&emsp;所以YOLO的核心思想是，直接在输出层回归bounding box的位置和bounding box所属的类别（整张图作为网络的输入，把 Object Detection 的问题转化成一个 Regression 问题）。YOLOv1 算法省掉了生成候选框的步骤，直接用一个检测网络进行端到端的检测，因而检测速度非常快。

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/YOLOv1/0.png)



## 二. 主要内容

#### 1. 模型架构

&emsp;&emsp;网络架构如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/YOLOv1/1.png)

&emsp;&emsp;设计 YOLOv1 的网络结构时参考了 GoogLeNet 的网络结构，但并未使用 Inception 的通道组合策略，而是大量使用了 1x1 和 3x3 卷积。前 24 层卷积用来提取图像特征，后面 2 层全连接用来预测目标位置和类别概率。在训练时先利用 ImageNet 分类数据集对前 20 层卷积层进行预训练，将预训练结果再加上剩下的四层卷积以及 2 层全连接，采用了 Leaky Relu 作为激活函数，其中为了防止过拟合对全连接层加了失活概率为 0.5 的 dropout 层。

&emsp;&emsp;从图中可以看到，yolo网络的输出的网格是7x7大小的（自注：对于输入448x448，相当于下采样到原来的1 / 64），另外，输出的channel数目为30。一个cell内，前20个元素是类别概率值，然后2个元素是边界框confidence，最后8个元素是边界框的 (x, y, w, h) 。

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/YOLOv1/2.png)

&emsp;&emsp;也就是说，每个cell有两个predictor，每个predictor分别预测一个bounding box的x，y，w，h和相应的confidence。但分类部分的预测却是共享的。**因此同一个cell内是没办法预测多个目标的**。（注：如果一个cell要预测两个目标，那么这两个predictor要怎么分工预测这两个目标？谁负责谁？不知道，所以没办法预测。**而像faster rcnn这类算法，可以根据anchor与ground truth的IOU大小来安排anchor负责预测哪个物体，所以后来yolo2也采用了anchor思想，同个cell才能预测多个目标**）



#### 2.模型输出的意义

**Confidence预测**

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/YOLOv1/3.png)

一个问题：在测试阶段，输出的confidece怎么算？还是通过Pr(object)和IOU计算吗？可是如果这样的话，测试阶段根本没有ground truth，那怎么计算IOU？

实际上，在测试阶段，网络只是输出了confidece这个值，但它已经包含了 Pr(object) * IOU ，并不需要分别计算Pr(object) 和 IOU（也没办法算）。为什么？因为你在训练阶段你给confidence打label的时候，给的是 Pr(object) * IOU 这个值，你在测试的时候，网络吐出来的也就是这个值。  



**Bounding box预测**

bounding box的预测包括xywh四个值。xy表示bounding box的中心相对于cell左上角坐标偏移，宽高则是相对于整张图片的宽高进行归一化的。偏移的计算方法如下图所示。

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/YOLOv1/4.png)

举例：比如S=10即一共10 x 10个grid，原图的宽高均为70pixels，加入某个x的坐标是38.5，则有38.5 x 10 / 70 - 5 = 0.55，可以看到恰好是归一化后的x坐标，简单推导一下就能看出来，原来坐标35的x是0.5，原来坐标42的x是0.6。

xywh为什么要这么表示呢？实际上经过这么表示之后，x，y，w，h都归一化了，它们的值都是在0 ~ 1之间。我们通常做回归问题的时候都会将输出进行归一化，否则可能导致各个输出维度的取值范围差别很大，进而导致训练的时候，网络更关注数值大的维度。因为数值大的维度，算loss相应会比较大，为了让这个loss减小，那么网络就会尽量学习让这个维度loss变小，最终导致区别对待。

坐标归一化各参数含义详解如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/YOLOv1/4_0.png)





**类别预测**

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/YOLOv1/5.png)

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/YOLOv1/5_0.png)



#### 3、训练

loss如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/YOLOv1/6.png)

将上述损失函数拆解如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/YOLOv1/6_0.png)

三者的损失类型均为均方损失函数，不同是在三部分损失前均有相应的权重系数来衡量三者各自的重要性。

关于loss，需要特别注意的是需要计算loss的部分。并不是网络的输出都算loss，具体地说：

1. 有物体中心落入的cell，需要计算分类loss，两个predictor都要计算confidence loss，预测的bounding box与ground truth IOU比较大的那个predictor需要计算x，y，w，h的loss。
2. 特别注意：没有物体中心落入的cell，只需要计算confidence loss。

另外，我们发现每一项loss的计算都是L2 loss，即使是分类问题也是。所以说yolo是把分类问题转为了回归问题。



## 三. 实验结果

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/YOLOv1/7.png)



## 四. 结论

* YOLOv1优点：1.速度快  2.在检测物体时能很好的利用上下文信息，不容易在背景上预测出错误的物体信息
* YOLOv1缺点：1.容易产生物体的定位错误。2.对小物体的检测效果不好（尤其是密集的小物体，因为同一cell只能预测2个物体，且不能预测同一类物体）。
