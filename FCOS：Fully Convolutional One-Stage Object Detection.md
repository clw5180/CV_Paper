# FCOS：Fully Convolutional One-Stage Object Detection

论文地址：https://arxiv.org/abs/1904.01355

代码复现：https://github.com/tianzhi0549/FCOS/



## 一、介绍

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FCOS/0.png)

简要概述文章精华
FCOS算法也是一篇anchor free的目标检测算法，但是其思想跟CornerNet系列有点不太一样，CornerNet系列的核心思想是通过Corner pooling来检测角点，然后对角点进行配对，最终得到检测结果，而FCOS方法借鉴了FCN的思想，对每个像素进行直接预测，预测的目标是到bounding box的上、下、左、右边的距离，非常的直观，另外为了处理gt重合的的时候，无法准确判断像素所属类别，作者引入了FPN结构，利用不同的层来处理不同的目标框，另外为了减少误检框，作者又引入了Center-ness layer，过滤掉大部分的误检框。FCOS的主干结构采用的是RetinaNet结构。

参考：https://blog.csdn.net/chunfengyanyulove/article/details/95091061



## 二、主要内容









## 三、实验结果

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FCOS/result0.png)

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FCOS/result1.png)

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FCOS/result2.png)



## 四、结论

* 
* 

&emsp;