# **ResNeXt：Aggregated Residual Transformations for Deep Neural Networks**

论文地址：<https://arxiv.org/abs/1611.05431>

代码复现：<https://github.com/facebookresearch/ResNeXt>



## 一、主要内容

&emsp;&emsp;”[ResNeXt](https://arxiv.org/abs/1611.05431) is a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, which we call “cardinality” (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width.“

即，ResNeXt是一个简单的、高度模块化的网络结构，用于图像分类任务。该架构依然采用堆叠构建块的方式构建。构建块内部采用分支结构，**分支的数目**称为 “**基数**”。作者认为，增加分支的数量比增加深度、宽度更高效，实验中也证明了 ResNeXt 的性能超过 ResNet。对于模型的表示方法，比如ResNeXt-50 (32x4d)，32就是所谓的基数cardinality，4d表示baseWidth=4。ResNeXt 在 ILSVRC 2016 分类比赛中获第二名，并且在ILSVRC 2017胜出的模型 **SENet architecture**也是以 **ResNeXt-152 (64 x 4d)** 为基础实现的。

&emsp;&emsp;VGG、ResNet 采用了堆叠相同构建块来构建网络。Inception 对网络的组件进行精心设计，从而在更低的计算量取得较高的准确率。Inception 有一个核心逻辑：split-transform-merge。虽然 Inception 的解空间是 大卷积层的解空间的子空间，但我们期待使用 split-transform-merge 策略去接近大卷积、dense层的表示能力。经过对组件精心的设计，Inception 的性能很高，但怎么去针对新数据集调整 Inception 的各个模块呢？所以作者提出了 ResNeXt，它采用 VGG / ResNet 类似的堆叠方式，同时以一种简单，可扩展的方式实现了 Inception 中的 split-transform-merge 策咯。（结构如图 1 右） 

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/ResNeXt/1_1.png)

#### 相关工作
- 多分支卷积网络： Inception 就是精心设计的多分支结构。ResNet 可以被看作一个两分支结构，一个分支是 identity mapping。深度神经决策森林是树状多分支网络，学习多个分离的函数。
- 分组卷积： 分组卷积最早可以追溯到 AlexNet。AlexNet 中分组卷积主要是为了用两块 GPU 来分布式训练。分组卷积的一个特例就是 Channel-wise 卷积。
- 压缩卷积网络： 卷积分解（在空间 and/or 通道层面）是一种常用的卷积网络冗余、加速、压缩网络的常用技术。相比于压缩，作者希望有更强的表示能力。
- 多模型集成： 对多个单独训练的网络进行平均是一种提高准确率的办法（在识别类比赛中广泛使用）。因为ResNet采用 additive behaviors，有人将 ResNet 理解为 一系列浅层网络 的集成。作者采用 加法 来聚合一系列的变换。但是作者认为将 ResNeXt 看作集成是不准确的，因为各个分支是同时训练的。

#### 模型结构

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/ResNeXt/1_2.png)

主要遵从了两个原则：

- feature map 大小不变时，标准堆叠
- 当 feature map 的大小减半，则通道数增加一倍

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/ResNeXt/1_3.png)

上面展示了 ResNeXt 的两个等价形式 

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/ResNeXt/1_4.png)

#### 模型容量

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/ResNeXt/1_5.png)

上面几个模型的 参数量 和 计算量 接近，基于此，作者认为两个模型的模型容量相近。




## 二、实验结果

**基数 vs. 宽度** 
在保持计算量不变的情况下，增大基数，能够减少误差。并且 ResNeXt 的训练误差更低，说明 表示能力更强。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/ResNeXt/2_0.png)



ResNeXts和DenseNets的对比（*DenseNet cosine is DenseNet trained with cosine learning rate schedule.*）：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/ResNeXt/2.png)



&emsp;&emsp;ResNet-50 到 ResNeXt-101 (64x4d)，不同模型的Top-1 Error如下图。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/ResNeXt/3.png)

在Detectron官网的MODEL_ZOO.md给出了R-50-C4到X-101-32x8d-FPN的结果（Fast & Mask R-CNN Baselines Using Precomputed RPN Proposals）：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/ResNeXt/4.png)

（一些名词解释：

- **Training Schedules**，即训练计划，这里采用了3种不同的方式，由 **lr schd** 这个列标识；
- **1x**: For minibatch size 16, this schedule starts at a LR of 0.02 and is decreased by a factor of * 0.1 after 60k and 80k iterations and finally terminates at 90k iterations. This schedules results in 12.17 epochs over the 118,287 images in `coco_2014_train` union `coco_2014_valminusminival` (or equivalently, `coco_2017_train`).
- **2x**: Twice as long as the 1x schedule with the LR change points scaled proportionally.
- **s1x** ("stretched 1x"): This schedule scales the 1x schedule by roughly 1.44x, but also extends the duration of the first learning rate. With a minibatch size of 16, it reduces the LR by * 0.1 at 100k and 120k iterations, finally ending after 130k iterations.

另外还有几个trick：

- All training schedules also use a 500 iteration linear learning rate warm up.
- Training uses multi-scale jitter over scales {640, 672, 704, 736, 768, 800}



## 三、结论

* 增加基数比深度、宽度更高效；
* 残差很有必要，如果没有残差效果显著下降；
