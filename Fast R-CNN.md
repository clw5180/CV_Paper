# Fast R-CNN（包括之前的R-CNN、SPP-Net）

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

黑色图片代表卷积之后的特征图，接着我们以不同大小的块来提取特征，分别是4x4，2x2，1x1，将这三张网格放到下面这张特征图上，就可以得到16+4+1=21种不同的**块（Spatial bins）**，我们从这21个块中，每个块提取出一个特征，这样刚好就是我们要提取的21维特征向量。这种以不同的大小格子的组合方式来池化的过程就是**空间金字塔池化（SPP）**。比如，要进行空间金字塔最大池化，其实就是从这21个图片块中，分别计算每个块的最大值，从而得到一个输出单元，最终得到一个21维特征的输出。以上图为例，**一共可以输出（16+4+1）x 256的特征**。

简而言之，即是将任意尺寸的feature map用三个尺度的金字塔层分别池化，将池化后的结果拼接得到固定长度的特征向量（图中的256为filter的个数），送入全连接层进行后续操作。例如上图，所以Conv5计算出的feature map也是任意大小的，现在经过SPP之后，就可以变成固定大小的输出了。

后来的Fast RCNN网络即借鉴了SPP的思想。SPP是pooling成多个固定尺度，再进行拼接，而RoI pooling只pooling到单个固定的尺度 。**多尺度学习能提高一点点mAP，不过计算量成倍的增加**。**对于“[SPP-net中的SPP pooling为什么还没有Fast R-CNN的ROI pooling效果好](https://blog.csdn.net/WYXHAHAHA123/article/details/86163140)” 的回答 —— spp不支持反向传播，spp那个时候，由于时代的局限性，没有backward，不能更新卷积层。是到了ROI pooling之后才做出backward的**。

另外注意，**R-CNN和SPP-net在训练时pipeline都是隔离的：提取proposal，CNN提取特征，SVM分类，bbox regression**。

以上参考：https://www.jianshu.com/p/884c2828cd8e



**3、Fast R-CNN的改进**：

（1）**training and testing end-to-end** ，这一点很重要，为了达到这一点其定义了ROI Pooling层，因为有了这个，使得训练效果提升不少。： 所有的特征都暂存在显存中，就不需要额外的磁盘空。
（2）**速度上的提升**，因为有了Fast R-CNN，这种基于CNN的 real-time 的目标检测方法看到了希望，在工程上的实践也有了可能，后续也出现了诸如Faster R-CNN/YOLO等相关工作。



## 二、主要内容

### SPP-Net论文解读

- #### 多尺度训练

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FastRCNN/5.png)

对于SPP-net的网络训练阶段：论文中将网络的训练分为两种：**一种是single-size,一种是Multi-size**。

（1）先讲解single-size的训练过程：理论上说，SPP-net支持直接以多尺度的原始图片作为输入后直接BP即可。实际上，caffe等实现中，为了计算的方便，GPU,CUDA等比较适合固定尺寸的输入，所以训练的时候输入是固定了尺度了的。**以224x224的输入为例**：

在conv5之后的特征图为：**13x13**（axa），即下采样16倍
金字塔层bins:   nxn
将pooling层作为sliding window pooling。
windows_size=[a/n] 向上取整 ， stride_size=[a/n]向下取整。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FastRCNN/6.png)

对于pool 3x3:      sizeX=5 的计算公式是：[13/3]向上取整=5 ，stride = 4的计算公式是：[13/3]向下取整。

如果输入改成180x180，这时候conv5出来的reponse map为10x10，类似的方法，能够得到新的pooling参数。

（2）**对于Multi-size training即就是：使用两个尺度进行训练：224x224 和180x180，预测的时候还是使用224**。

自注：另外有一点作者提到，如果尺度在[180，224] 之间随机采样，那么SPP-net（Overfeat-7）的top-1和top-5的分类误差比只有180和224两个尺度训练还要高一些（30.06%/10.96% ，高于29.68%/? ），作者认为原因可能是如果是区间的话，224这个尺度出现的次数会更少（“because the size of 224 (which is used for
testing) is visited less ”）。

**训练的时候，224x224的图片通过crop得到，180x180的图片通过缩放224x224的图片得到。之后，迭代训练，即用224的图片训练一个epoch，之后180的图片训练一个epoch，交替地进行**。两种尺度下，**输出的特征维度都是(9+4+1)x256，参数是共享的**，之后接全连接层即可。论文中说，这样训练的好处是可以**更快地收敛**。



- #### Mapping a Window to Feature Maps

我们知道，在原图中的proposal,经过多层卷积之后，位置还是相对于原图不变的（如下图所示），那现在需要解决的问题就是，如何能够将原图上的proposal,映射到卷积之后得到的特征图上，因为在此之后我们要对proposal进行金字塔池化。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FastRCNN/7.png)

对于映射关系，论文中给出了一个公式：

假设(x’,y’)表示特征图上的坐标点，坐标点(x,y)表示原输入图片上的点，那么它们之间有如下转换关系，这种映射关心与网络结构有关： (x,y)=(S*x’,S*y’)

反过来，我们希望通过(x,y)坐标求解(x’,y’)，那么计算公式如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FastRCNN/8.png)



- #### 多尺度预测

对于检测算法，论文中是这样做到：使用ss生成~2k个候选框，缩放图像min(w,h)=s之后提取特征，每个候选框使用一个4层的空间金字塔池化特征，网络使用的是ZF-5的SPPNet形式。之后将12800d的特征输入全连接层，SVM的输入为全连接层的输出。

这个算法可以应用到多尺度的特征提取：先将图片resize到五个尺度：480，576，688，864，1200，加自己6个。然后在map window to feature map一步中，选择ROI框尺度在｛6个尺度｝中大小最接近224x224的那个尺度下的feature maps中提取对应的roi feature。这样做可以提高系统的准确率。


参考：https://blog.csdn.net/v1_vivian/article/details/73275259



## 三、总结

用一张图来完整描述SPP-net的结构：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FastRCNN/9.png)