# **MobileNet v1：Efficient Convolutional Neural Networks for Mobile Vision Applications**

论文地址：<https://arxiv.org/abs/1704.04861>

代码复现：<https://github.com/marvis/pytorch-mobilenet>



## 一、背景

&emsp;&emsp;在某些真实的应用场景如移动或者嵌入式设备，如此大而复杂的模型是难以被应用的。首先是模型过于庞大，面临着**内存不足**的问题，其次这些场景要求低延迟，或者说**响应速度要快**，想象一下自动驾驶汽车的行人检测系统如果速度很慢会发生什么可怕的事情。所以，研究小而高效的CNN模型在这些场景至关重要。目前的研究总结来看分为两个方向：（1）是对训练好的复杂模型进行**压缩**得到小模型；（2）**直接设计小模型**并进行训练。本文的主角MobileNet属于后者，其是Google最近提出的一种小巧而高效的CNN模型，其在accuracy和latency之间做了折中。

&emsp;&emsp;MobileNet自从2017年由谷歌公司提出，MobileNet可谓是轻量级网络中的Inception，经历了一代又一代的更新。成为了学习轻量级网络的必经之路。



## 二、主要内容

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



![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/R2CNN/1.png)





## 三、实验结果





## 四、结论

* 
* 