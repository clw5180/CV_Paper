# **MobileNet v1：Efficient Convolutional Neural Networks for Mobile Vision Applications**

论文地址：<https://arxiv.org/abs/1704.04861>

代码复现：<https://github.com/marvis/pytorch-mobilenet>



## 一、背景

&emsp;&emsp;在某些真实的应用场景如移动或者嵌入式设备，如此大而复杂的模型是难以被应用的。首先是模型过于庞大，面临着**内存不足**的问题，其次这些场景要求低延迟，或者说**响应速度要快**，想象一下自动驾驶汽车的行人检测系统如果速度很慢会发生什么可怕的事情。所以，研究小而高效的CNN模型在这些场景至关重要。目前的研究总结来看分为两个方向：（1）是对训练好的复杂模型进行**压缩**得到小模型；（2）**直接设计小模型**并进行训练。本文的主角MobileNet属于后者，其是Google最近提出的一种小巧而高效的CNN模型，其在accuracy和latency之间做了折中。

&emsp;&emsp;MobileNet自从2017年由谷歌公司提出，MobileNet可谓是轻量级网络中的Inception，经历了一代又一代的更新。成为了学习轻量级网络的必经之路。


## 二、深度可分离卷积简介

#### - 空间可分离

顾名思义，空间可分离就是将一个大的卷积核变成两个小的卷积核，比如将一个3×3的核分成一个3×1和一个1×3的核：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/深度可分离卷积/1.png)

由于空间可分离卷积不在MobileNet的范围内，就不说了。

#### - 深度可分离卷积
![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/深度可分离卷积/2.png)

深度可分离卷积就是将普通卷积拆分成为一个**深度可分离卷积** (或称**深度卷积**，depthwise convolution) 和一个**逐点卷积** (pointwise convolution) 。


**标准卷积与深度可分离卷积的过程对比如下**：
![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/深度可分离卷积/3.jpg)

深度可分离卷积**用更少的参数，更少的运算，但是能达到和标准卷积差不多的结果**。计算一下标准卷积的参数量与计算量：
![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/深度可分离卷积/4.png)
我们通常所使用的是3×3的卷积核，也就是**参数和计算量会下降到原来的九分之一到八分之一**，但是**准确率只有下降极小的1％**，如下图所示。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/深度可分离卷积/5.png)

可以发现，作为轻量级网络的V1在计算量小于GoogleNet，参数量差不多是在一个数量级的基础上，在分类效果上比GoogleNet还要好，这就是要得益于深度可分离卷积了。VGG16的计算量参数量比V1大了30倍，但是结果也仅仅只高了1%不到。




参考：<https://www.zhihu.com/search?type=content&q=%20%E7%A9%BA%E9%97%B4%E5%8F%AF%E5%88%86%E7%A6%BB%E5%8D%B7%E7%A7%AF>



## 三、主要内容

&emsp;&emsp;MobileNet的基本单元是深度级**可分离卷积（depthwise separable convolution，dw）**，其实这种结构之前已经被使用在Inception模型中。Pytorch版本实现如下：

```python
def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
    
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
        )
```







## 三、实验结果





## 四、结论

* 
* 