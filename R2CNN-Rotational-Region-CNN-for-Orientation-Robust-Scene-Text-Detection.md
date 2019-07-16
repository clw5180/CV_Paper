

# **R2CNN-Rotational-Region-CNN-for-Orientation-Robust-Scene-Text-Detection**

论文发布日期：2017.6.29  [CVPR]
论文链接：https://arxiv.org/abs/1706.09579

## 1. Introduction  
&emsp;&emsp;文本检测的**特点和挑战** ：多尺度、不同的宽高比、不同字体风格、光照、透视畸变（不同的拍摄角度）以及方向等。对于场景文本检测（scene text detection），除了要预测轴向对齐（axis-aligned）的坐标信息之外，还要预测出文本的**方向**，这对于场景文本的识别十分重要。这篇文章提出了**R2CNN（Rotational Region CNN**）算法解决**旋转文本**的检测。R2CNN算法的整体结构如下图 ：


* 本文方案

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/1.png)

&emsp;&emsp;R2CNN采用的架构是两阶段的Faster RCNN，如上图所示。

&emsp;&emsp;第一个阶段：通过RPN网络，得到**水平框**的Proposal（图b）。由于很多文字是很小的，因此将Faster RCNN中的anchor scale(8, 16, 32)改为(4, 8, 16, 32)，实验证明增加了小尺度anchor后，检测效果明显提升。

&emsp;&emsp;第二个阶段：对于每个proposal，在RoI Pooling时平行地使用了不同的pooled size（之前只有7×7，现在额外增加了**3×11和11×3**），这种设置的用意也非常明显，就是为了**解决水平和竖直长文本的检测**。然后将提取到的RoI特征concat在一起，经过fc6，fc7，再进行**分类**预测（得到text/non-text的置信度，二分类）、**水平框**（axis-aligned box）预测、和**倾斜框**（inclined box）预测（图c）；之后经过斜框NMS进行后处理，得到最后的结果（图d）。

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/2.png)

&emsp;&emsp;综上，简单来说就是**首先RPN生成正常的水平proposal，二分类筛掉，然后保留的每个proposal回归水平框和斜框坐标，最后对斜框进行斜向NMS得到最终检测结果**。（注：R2CNN和RRPN的区别，**RRPN是直接在RPN阶段就会生成倾斜的矩形框**；而作者认为使用生成水平矩形框的RPN就已经足够了）

&emsp;&emsp;另外作者在本文中提到了目前主流的文本检测的深度学习的方法有：**TextBoxes**是一个具有单个深度神经网络的end to end的快速场景文本检测器。**DeepText**通过Inception-RPN生成单词的proposals，然后文本检测网络对每个单词的proposal进行位置精修并给出score。**FCRN**（全卷积回归网络）是利用合成图像训练场景文本检测模型。但是，上述这些方法都生成的是水平框（axis-aligned detection boxes），不能解决文本的方向问题。**CTPN**检测固定宽度的vertical boxes，然后使用BLSTM捕获顺序信息，然后连接vertical boxes以获得最终检测框；CTPN擅长检测水平方向的文本，但不是和检测有倾斜角度的文本。基于**FCN**（全卷积网络）用于检测多种方向的场景文本，需要以下3个步骤：一是文本块FCN的文本检测，二是基于MSER的多向文本行候选的生成，三是文本行候选的分类。**RRPN**也用于检测任意方向的场景文本。RPN生成具有倾斜角度的proposal，并且后面的分类和回归会基于这些proposal。**SegLink**提出通过检测段落和连接来检测具有特定方向的文本，可以用于检测任意长度的文本行。**EAST**的特点是快速和准确。**DMPNet**使用更紧密的四边形来检测文本。




## 2. Proposed Approach

### 2.1 Problem definition
* gt定义 
  &emsp;&emsp;文本检测的ICDAR竞赛数据集是不规则的四边形，用顺时针的四个点坐标进行表征，不是严格的矩形框。这种不规则四边形一般可以用斜矩形来近似拟合，所以后面的bbox都采用斜矩形进行bbox预测。

* 旋转表征 
  &emsp;&emsp;关于旋转方向的表征，很多方法是直接选择的角度回归来直接表征， 比如可以用Figure3(a)即下图所示的4个点坐标来表示一个任意形状的四边形，这种表示方式其实就覆盖了水平框和倾斜框，而且框的形状不仅限于矩形，而是延伸至四边形。但**这篇文章不采用任意形状的四边形预测方式，而认为倾斜矩形框足够覆盖待检测的文本**，因此这篇文章所提到的倾斜框是指倾斜的矩形框，**为了描述简单，后续都用倾斜框代替倾斜的矩形框**。倾斜框的定义也有好几种，比如可以用水平框+旋转角度来表示一个倾斜框，这种表示方式在旋转目标检测算法中也比较常见，比如EAST算法。但是这篇文章认为训练旋转角度在有些角度情况下模型不是很稳定，比如90度和-90度的矩形的外在表征很相近，但角度值相差很大，**所以考虑采用Figure3(b)(c)，即（x1, y1, x2, y2, h）这种用两个相邻点坐标和高度来定义一个倾斜框**。注意(x1, y1)位于左上角，（x2, y2）是从(x1, y1)顺时针方向的第一个经过的点。
  ![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/3.png)

但也有人认为，觉得(x1,y1,x2,y2,h)并不是表示斜矩形唯一的方式，只要能够描述斜矩形的方式都是可以的，只不过不同的表示方式可能会影响网络的学习。可以用中心点坐标，宽度，高度和旋转角度来表示斜矩形，对应的关系如下图：

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/3-1.jpg)



### 2.2 Rotational Region CNN (R2CNN)

* RPN for proposing axis-aligned boxes  
  &emsp;&emsp;RPN和常规的相似，加了更小的anchor尺寸以便获得更好的效果。其他配置均同Faster R-CNN。

* RoIPoolings of different pooled sizes  
  &emsp;&emsp;为了适应文本不同方向的特殊长宽比，除了标准7x7外，还采用了不同比例的RoIpooling为11x3和3x11的信息进行获取融合

* Regression for text/non-text scores, axis-aligned boxes, and inclined minimum area boxes  
  &emsp;&emsp;在回归上，<u>不仅回归斜框坐标，还回归正框</u>，实验表明这样的效果能提高。

* Inclined non-maximum suppression    
  &emsp;&emsp;实验表明，斜的NMS效果会比正的好。很好理解，对于<u>斜向密集目标的检测</u>，常规NMS很容易出现IoU过大而被抑制，漏检很大，斜向NMS就不会。

    

  &emsp;&emsp;**Figure4是关于倾斜框的NMS算法**。(a)是R2CNN算法的水平框和倾斜框预测结果合并在一张图上的结果。(b)是采用水平框的NMS算法处理(a)中的预测框后得到的结果，可以看到有部分正确的文本框被剔除掉（红色虚线）。(c)是采用倾斜框的NMS算法处理(a)中的预测框后得到的结果，可以看到结果比较好。基于倾斜框的NMS算法和传统的基于水平矩形框的NMS算法差别不大，只不过计算对象换成两个倾斜框，这个操作还是比较重要的。 

  ![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/4.jpg)



### 2.3 Training objective (Multi-task loss)

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/5.png)

损失函数方面从Figure2的网络结构图也可以看出一共包含3个部分：

1、有无文本的二分类损失Lcls。

2、水平框的回归损失Lreg(vi,vi*)。

3、倾斜框的回归损失Lreg(ui,ui*)。后面两部分都是采用目标检测中常用的smooth L1损失函数，所以损失函数方面没有太大的改动。 



## 3. Experiments
Table1是在IDCAR 2015数据集上不同参数配置时R2CNN测试结果。 

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/6.png)

一张表就够了，可以看出：    
1. 加入正框预测的效果涨点值得一看，应该是让他更好地学习    
2. 加入小尺度anchor效果不错    
3. 多尺度pooling的效果也还行，不是特别明显 



Table2是R2CNN算法和其他算法在IDCAR2015数据集上的对比结果。 

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/7.jpg)



Table3是R2CNN算法和其他算法在IDCAR2013数据集上的对比结果。

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/8.jpg)




## Conclusion
* 采用多尺度pooling提取不同宽高比的信息
* 斜框的NMS解决传统NMS的密集漏检问题
* 正斜框的MTL（RRPN的思路不同）
