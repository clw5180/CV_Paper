# **SSD: Single Shot MultiBox Detector**

论文地址：<https://arxiv.org/abs/1512.02325>

代码复现：https://github.com/weiliu89/caffe/tree/ssd （作者实现）



## 一、介绍

&emsp;&emsp;SSD是单阶段检测器。对于300x300的输入图片，可以在VOC2007上达到74.3%的mAP，并且速度高达59FPS；对于512x512更大尺寸的输入，SSD可以达到更高的76.9%的mAP，甚至超过了Faster R-CNN；在经过**改进的数据增强方法**后，可以分别达到77.2%和79.8%的mAP，可以看到**提高了3个点左右**。相对于其他单阶段监测器，SSD可以在较小的输入尺寸上得到较大的精确度。在速度上的提高主要是通过去掉了bbox的候选框，以及去掉了特征重采样的部分；

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/SSD/1.png)

&emsp;&emsp;SSD使用了多尺度feature map，如上图，狗比猫的尺度更大，在8x8的特征图中不能检测出大狗，但能检测出猫，而在4x4的特征图中就可以检测出狗，却不能检测出猫。这样，不同尺度的目标就可以通过不同层的特征图检测出。



## 二、主要内容





## 三、实验结果

&emsp;&emsp;数据增强十分重要。在Faster R-CNN中只使用了水平翻转，而SSD中使用了更为强大的采样策略，类似于YOLO，使最终的mAP提升了接近9%。数据增强对于小物体效果显著；

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/SSD/5.png)

## 四、结论

* 
* 
