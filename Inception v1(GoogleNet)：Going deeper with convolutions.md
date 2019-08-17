# **Inception v1(GoogleNet)：Going deeper with convolutions**

论文地址：<https://arxiv.org/pdf/1409.4842.pdf>

代码复现：



## 一、介绍

&emsp;&emsp;在提出Inception v1或称GoogleNet之前，网络大都是类似LeNet的卷积层和池化层的顺序连接。这样的话，要想提高精度，增加网络深度和宽度是一个有效途径，但也面临着参数量过多、过拟合等问题。有没有可能在同一层就可以提取不同（稀疏或不稀疏）的特征呢(使用不同尺寸的卷积核)？于是，2014年，在其他人都还在一味的增加网络深度时(比如vgg)，GoogleNet就率先提出了**卷积核的并行合并（也称Bottleneck Layer**），这样的结构主要有以下改进：（1）一层block就包含1x1卷积，3x3卷积，5x5卷积，3x3池化(使用这样的尺寸不是必需的，可以根据需要进行调整)。这样，网络中每一层都能**学习到“稀疏”（3x3、5x5）或“不稀疏”（1x1）的特征，既增加了网络的宽度，也增加了网络对尺度的适应性**； （2）通过**DepthConcat**在每个block后合成特征，获得**非线性属性**；这是一种聚合操作，**在输出通道这个维度上聚合**（一个inception module每个分支通道数可能不一样，但是feature map大小应该是一样的。strides=1，padding=same）。从inception 3a层开始引入，比如该层输入：28×28×192；输出：由于每个分支strides=1，padding=same，所以只是通道数在变化，feature map大小不变。**最终输出 28×28×256（只增加了少量通道数）**。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/GoogleNet/0.png)

&emsp;&emsp;注意到上面右图中，GooLeNet借鉴Network-in-Network的思想，**使用1x1的卷积核实现降维操作，减少输出通道数量**（也间接增加了网络的深度），以此来减小网络的参数量；另外用**全局平均池化层代替了全连接**（VGG中全连接层的参数占据了90%的参数量） 也可以减小网络的参数量。

&emsp;&emsp;GoogleNet首次出现在ILSVRC 2014比赛中（和VGG同年），获得了当时比赛的第一名。使用了Inception的结构，当时比赛的版本叫做Inception V1。inception结构现在已经更新了4个版本。Going deeper with convolutions这篇论文就是指的Inception V1版本。控制了计算量和参数量的同时，获得了很好的分类性能。**500万的参数量只有AlexNet的1/12（6000万）**。



## 二、主要内容

GoogleNet有22层深，比同年的VGG19还深。包含了9个inception module，下面是具体的结构。在整个网络中，会有多个堆叠的inception module，**希望靠后的inception module可以捕捉更高阶的抽象特征，因此靠后的inception module中，大的卷积应该占比变多**。

最后实现的inception v1网络是上图结构的顺序连接，如下图所示。注意到网络中**含多个辅助分类器**：Google net除了最后一层输出进行分类外，其中间节点的分类效果也很好。于是，**GoogleNet也会将中间的某一层的输出用于分类**，并**按一个较小的权重（0.3）**加到最终的分类结果中。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/GoogleNet/1.png)

其中不同inception模块之间使用2x2的最大池化进行下采样，如下表所示。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/GoogleNet/2.png)





## 三、实验结果





## 四、结论

* 
* 



可参考：

<https://zhuanlan.zhihu.com/p/33075914>