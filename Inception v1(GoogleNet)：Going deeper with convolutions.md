# **Inception v1(GoogleNet)：Going deeper with convolutions**

论文地址：<https://arxiv.org/pdf/1409.4842.pdf>

代码复现：



## 一、介绍

&emsp;&emsp;GoogleNet首次出现在ILSVRC 2014比赛中（和VGG同年），获得了当时比赛的第一名。使用了Inception的结构，当时比赛的版本叫做Inception V1。inception结构现在已经更新了4个版本。Going deeper with convolutions这篇论文就是指的Inception V1版本。控制了计算量和参数量的同时，获得了很好的分类性能。**500万的参数量只有AlexNet的1/12（6000万）**。











## 二、主要内容

GoogleNet有22层深，比同年的VGG19还深。包含了9个inception module，下面是具体的结构。在整个网络中，会有多个堆叠的inception module，**希望靠后的inception module可以捕捉更高阶的抽象特征，因此靠后的inception module中，大的卷积应该占比变多**。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/GoogleNet/1.png)

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/GoogleNet/2.png)



#### 主要关注网络的以下几个特点：

- **DepthConcat：**聚合操作，在输出通道这个维度上聚合（一个inception module每个分支通道数可能不一样，但是feature map大小应该是一样的。strides=1，padding=same）。从inception 3a层开始引入，比如该层输入：28×28×192；输出：由于每个分支strides=1，padding=same，所以只是通道数在变化，feature map大小不变。最终输出 28×28×256（只增加了少量通道数）。

- **含多个辅助分类器**：Google net除了最后一层输出进行分类外，其中间节点的分类效果也很好。于是，**GoogleNet也会将中间的某一层的输出用于分类**，并**按一个较小的权重（0.3）**加到最终的分类结果中。

- **降低了参数量**，主要通过：（1）用全局平均池化层代替了全连接（VGG中全连接层的参数占据了90%的参数量） （2）大量1×1的卷积核的使用，可以降低通道数量，从而减少了大量参数。

下面两图对比了使用1x1卷积前后的inception module结构：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/GoogleNet/3.png)

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/GoogleNet/4.png)





## 三、实验结果





## 四、结论

* 
* 



可参考：

<https://zhuanlan.zhihu.com/p/33075914>