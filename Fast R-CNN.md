# Fast R-CNN

论文地址：https://arxiv.org/abs/1504.08083



## 一、介绍

先回顾一下 R-CNN 和 SPP-net：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FastRCNN/1.jpg)

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FastRCNN/2.jpg)



**1、R-CNN**：RCNN可以看作是RegionProposal+CNN这一框架的开山之作，在imgenet/voc/mscoco上基本上所有top的方法都是这个框架，可见其影响之大。**R-CNN的主要缺点是重复计算**，后来MSRA的kaiming组的SPPNET做了相应的加速。



**2、SPP-net（Spatial Pyramid Pooling）**：spp提出的**初衷是为了解决CNN对输入图片尺寸的限制**。由于全连接层的存在，与之相连的最后一个卷积层的输出特征需要固定尺寸，从而要求输入图片尺寸也要固定。R-CNN及之前的做法是将图片裁剪或变形（crop/warp）；crop/warp的一个问题是导致图片的信息缺失或变形，影响识别精度。对此，文章中在最后一层卷积特征图的基础上又进一步进行处理，提出了spatial pyramid pooling，如下图所示。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FastRCNN/3.png)

空间金字塔池化（spatial pyramid pooling）的网络结构如下图：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FastRCNN/4.png)

简而言之，即是将任意尺寸的feature map用三个尺度的金字塔层分别池化，将池化后的结果拼接得到固定长度的特征向量（图中的256为filter的个数），送入全连接层进行后续操作。

后来的Fast RCNN网络即借鉴了SPP的思想。SPP是pooling成多个固定尺度，再进行拼接，而RoI pooling只pooling到单个固定的尺度 。**多尺度学习能提高一点点mAP，不过计算量成倍的增加**。**对于“[SPP-net中的SPP pooling为什么还没有Fast R-CNN的ROI pooling效果好](https://blog.csdn.net/WYXHAHAHA123/article/details/86163140)” 的回答 —— spp不支持反向传播，spp那个时候，由于时代的局限性，没有backward，不能更新卷积层。是到了ROI pooling之后才做出backward的**。

另外注意，**R-CNN和SPP-net在训练时pipeline都是隔离的：提取proposal，CNN提取特征，SVM分类，bbox regression**。

以上参考：https://www.jianshu.com/p/884c2828cd8e



**3、Fast R-CNN的改进**：

（1）**training and testing end-to-end** ，这一点很重要，为了达到这一点其定义了ROI Pooling层，因为有了这个，使得训练效果提升不少。： 所有的特征都暂存在显存中，就不需要额外的磁盘空。
（2）**速度上的提升**，因为有了Fast R-CNN，这种基于CNN的 real-time 的目标检测方法看到了希望，在工程上的实践也有了可能，后续也出现了诸如Faster R-CNN/YOLO等相关工作。



## 二、主要内容









## 三、实验结果





## 四、结论

* 
* 

&emsp;