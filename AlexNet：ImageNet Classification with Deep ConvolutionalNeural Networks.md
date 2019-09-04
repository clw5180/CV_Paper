# **AlexNet：ImageNet Classification with Deep Convolutional Neural Networks**

论文地址：http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

代码复现：http://caffe.berkeleyvision.org/gathered/examples/imagenet.html



## 一、介绍

&emsp;&emsp;训练集1.2million张，验证集50000张，测试集150000张。对于ImageNet LSVRC-2010，top-5 error rate为17%，top-1 error rate为37.5%；ImageNet ILSVRC-2012的top-5 error rate为15.3%。**网络参数**：**60 million parameters** and 650,000 neurons；



## 二、主要内容

&emsp;&emsp;首先将图片resize到256x256，因此对于长方形的图片，把短边resize到256,然后crop得到多个256x256的图片（实际输入网络的是224x224）。预处理阶段只是减去了RGB的均值。



网络结构如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/AlexNet/1.png)

包括8个需要学习的层：其中5个CNN层和3个全连接层。

为解决sigmoid和tanh在饱和区域梯度下降缓慢的问题，引入ReLU激活函数f(x) = max(0, x)，训练速度比同等条件下的tanh快了6倍。



使用2块GTX 580（3GB）进行训练





## 三、实验结果





## 四、结论

* 
* 
