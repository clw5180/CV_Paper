论文：EAST：An Efficient and Accurate Scene Text Detector
============

论文地址：https://arxiv.org/pdf/1704.03155.pdf

代码复现：https://github.com/argman/EAST



## 1. 介绍

&emsp;&emsp;传统的文本检测方法和一些基于深度学习的文本检测方法，大多是multi-stage，在训练时需要对多个stage调优，这势必会影响最终的模型效果，而且非常耗时．针对上述存在的问题，本文提出了端到端的文本检测方法，消除中间多个stage(如候选区域聚合，文本分词，后处理等)，直接预测文本行．



#### 论文关键idea

- 提出了基于two-stage的文本检测方法：全卷积网络(FCN)和非极大值抑制(NMS)，消除中间过程冗余，减少检测时间．
- 该方法即可以检测单词级别，又可以检测文本行级别．检测的形状可以为任意形状的四边形：即可以是旋转矩形(下图中绿色的框)，也可以是普通四边形(下图中蓝色的框)）．
- 采用了Locality-Aware NMS来对生成的几何进行过滤
- 该方法在精度和速度方面都有一定的提升．



## 2. 主要内容

#### Pipeline

论文的思想非常简单，结合了DenseBox和Unet网络中的特性，具体流程如下：

- 先用一个通用的网络(论文中采用的是Pvanet，实际在使用的时候可以采用VGG16，Resnet等)作为base net ，用于特征提取
- 基于上述主干特征提取网络，抽取不同level的feature map（它们的尺寸分别是inuput-image的 ![\frac{1}{32}，\frac{1}{16}，\frac{1}{８}，\frac{1}{４}](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B32%7D%EF%BC%8C%5Cfrac%7B1%7D%7B16%7D%EF%BC%8C%5Cfrac%7B1%7D%7B%EF%BC%98%7D%EF%BC%8C%5Cfrac%7B1%7D%7B%EF%BC%94%7D) ），这样可以得到不同尺度的特征图．目的是解决文本行尺度变换剧烈的问题，ealy stage可用于预测小的文本行，late-stage可用于预测大的文本行．
- 特征合并层，将抽取的特征进行merge．这里合并的规则采用了U-net的方法，合并规则：从特征提取网络的顶部特征按照相应的规则向下进行合并，这里描述可能不太好理解，具体参见下述的网络结构图
- 网络输出层，包含文本得分和文本形状．根据不同文本形状(可分为RBOX和QUAD)，输出也各不相同，具体参看网络结构图

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/1.png)

####  实现细节
#####  - 合并层中feature map的合并规则

具体的合并步骤如下：

1. 特征提取网络层中抽取的最后层feature map被最先送入uppooling层(这里是将图像放大原先的２倍)，
2. 然后与前一层的feature map进行concatenate，
3. 接着依次送入卷积核大小为3x3，卷积核的个数随着层递减,依次为128，64，32
4. 重复1-3的步骤２次
5. 将输出经过一个卷积核大小为3\times{3}3×3，核数为32个

具体的公式如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/2.png)

##### - 输出层的输出

- 对于检测形状为RBOX，则输出包含文本得分和文本形状(AABB boundingbox 和rotate angle)，也就是一起有６个输出，这里AABB分别表示相对于top,right,bottom,left的偏移
- 对于检测形状为QUAD，则输出包含文本得分和文本形状(８个相对于corner vertices的偏移)，也就是一起有９个输出，其中QUAD有８个，分别为 (x1, y1, x2, y2, x3, y3, x4, y4)

##### -  训练标签生成
- QUAD的分数图生成与几何形状图生成（暂略，详见<https://zhuanlan.zhihu.com/p/37504120>）

##### - 训练loss
loss由两部分组成：score map loss 和geometry loss,具体公式如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/3.png)

- 分数图损失(score map loss)
论文中采用的是类平衡交叉熵，用于解决类别不平衡训练，避免通过 平衡采样和硬负挖掘 解决目标物体的不不平衡分布，简化训练过程，具体公式如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/4.png)

但是在具体实战中，一般采用dice loss，它的收敛速度会比类平衡交叉熵快

几何形状损失(geometry loss)
文本在自然场景中的尺寸变化极大。直接使用L1或者L2损失去回归文本区域将导致损失偏差朝更大更长．因此论文中采用IoU损失在RBOX回归的AABB部分，尺度归一化的smoothed-L1损失在QUAD回归，来保证几何形状的回归损失是尺度不变的．

针对RBOX loss，其损失函数公式为：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/5.png)

- 局部感知NMS(locality-aware NMS)
与通用目标检测相似，阈值化后的结果需要经过非极大值抑制(NMS)来得到最终的结果．由于本文中面临的是成千上万个几何体，如果用普通的NMS，其计算复杂度是O(n^2)O(n2)，n是几何体的个数，这是不可接受的．

针对上述时间复杂度问题，本文提出了基于行合并几何体的方法，当然这是基于邻近几个几何体是高度相关的假设．注意：这里合并的四边形坐标是通过两个给定四边形的得分进行加权平均的，也就是说这里是**"平均"而不是"选择"几何体**．

- 训练其它参数
  整个训练采用基于Adam的端到端的训练．为了加速学习，统一从图像集中采样512*512构建minibatch大小为24，然后staged learning rate decay，但是在具体实验过程中可以使用linear learning rate decay．



## 3. 训练结果

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/6.png)



## 4. 总结

- 在特征合并层，利用不同尺度的feature map，并通过相应的规则进行自顶向下的合并方式，可以检测不同尺度的文本行
- 提供了文本的方向信息，可以检测各个方向的文本
- 本文的方法在检测长文本的时候效果表现比较差，这主要是由网络的感受野决定的(感受也不够大)
- 在检测曲线文本时，效果不太理想