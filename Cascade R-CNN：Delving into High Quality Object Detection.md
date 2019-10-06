# **Cascade R-CNN: Delving into High Quality Object Detection**

论文地址：https://arxiv.org/abs/1712.00726

代码复现：https://github.com/zhaoweicai/Cascade rcnn



delving into 深入研究，钻研细节

paradox 矛盾的人或事，悖论



## 一、介绍

对一个两阶段的目标检测器而言，如果IoU threshold=0.5，通常会产生较多误检，因为0.5的阈值会使得正样本中有较多的背景，要想训练出一个能够排除掉这类close false positive的检测器（即减少FP的发生）是比较难的；如果IoU threshold=0.7，通常会由于正样本数量过少导致过拟合。Cascade R-CNN的提出主要就是解决这个问题，**简单讲Cascade R-CNN是由一系列的检测模型组成，每个检测模型都基于不同IoU阈值的正负样本训练得到，前一个检测模型的输出作为后一个检测模型的输入，因此是stage by stage的训练方式，而且越往后的检测模型，其界定正负样本的IoU阈值是不断上升的**，通过不断提高一系列stage在训练时的IoU threshold来处理容易产生FP（close false positives）的样本。



## 二、主要内容

Figure1（c）和（d）中的曲线是用来描述localization performance，其中横坐标表示输入proposal和ground truth的IoU值，纵坐标表示输出的proposal和ground truth的IoU值。红、绿、蓝3条曲线代表训练检测模型时用的正负样本标签的阈值分别是u=0.7、0.6、0.5；从（c）可以看出，比如对于Faster R-CNN的RPN网络，当采用某个阈值（假设u=0.5，u即IoU threshold）来界定正负样本时，如果多数anchor和gt的IoU都在0.5附近，那么训练时采用u=0.5的模型就比u=0.6的模型好；如果多数anchor和gt的IoU都高于0.5不少，比如在0.75附近，那么训练时采用u=0.5的模型就不如u=0.6的模型。即作者所说：“In general, a detector optimized at a single IoU level is not necessarily optimal at other levels”。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/CascadeRCNN/1.png)

从Figure1（c）可以看出，用不同的IoU阈值训练得到的检测模型对和gt有着不同IoU的输入proposal的效果差别较大，因此**希望训练时模型用的IoU阈值要尽可能和输入proposal的IoU接近**。另一方面：可以看Figure1（c）中的曲线，三条彩色曲线基本上都在灰色曲线以上，这说明对于这三个阈值而言，**输出IoU基本上都大于输入IoU**。那么就可以以上一个stage的输出作为下一个stage的输入，这样就能得到越来越高的IoU。总之，**很难让一个在指定IoU阈值界定的训练集上训练得到的检测模型对IoU跨度较大的proposal输入都达到最佳，因此采取Cascade的方式能够让每一个stage的detector都专注于检测IoU在某一范围内的proposal，因为输出IoU普遍大于输入IoU，因此检测效果会越来越好**。 **因此，考虑设计成Cascade R-CNN这种级联结构**。（自注：个人感觉应该是最前面RPN的输入还是RPN的anchor，然后后输出u>0.5的proposal作为下一级（stage2）的输入，下一级就不再需要RPN了，而是直接用stage1的proposal当做anchor，然后输出u>0.6的proposal作为下一级（stage3）的输入，最后输出u>0.7的proposals，这样得到的正样本数比直接用一个阈值为0.7的RPN多很多，因为很多都是IoU接近0.5的proposal通过一次或两次回归最终与gt的IoU超过0.7，被视为正样本。正样本数量多了，比如有的是框出来偏左上角的位置，有的是偏右上角的位置等等各种不同的与gt的IoU超过0.7的proposals，这样模型的泛化能力得到很大提升。具体过程详见下图）

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/CascadeRCNN/2.png)



Figure3是关于几种网络结构的示意图。（a）是Faster R-CNN，因为two stage类型的object detection算法基本上都基于Faster R-CNN，所以这里也以该算法为基础算法。（b）是迭代式的bbox回归，从图也非常容易看出思想，就是前一个检测模型回归得到的bbox坐标初始化下一个检测模型的bbox，然后继续回归，这样迭代三次后得到结果。（c）是Integral Loss，表示对输出bbox的标签界定采取不同的IOU阈值，因为当IOU较高时，虽然预测得到bbox很准确，但是也会丢失一些bbox。（d）就是本文提出的Cascade R-CNN。Cascade R-CNN看起来和（b）这种迭代式的bbox回归以及（c）这种Integral Loss很像；对于和（b）的不同点，作者在论文中给出：“First, while iterative BBox is a post-
processing procedure used to improve bounding boxes, cascaded regression is a resampling procedure that changes the distribution of hypotheses to be processed by the different stages. Second, because it is used at both training and inference, there is no discrepancy between training and inference distributions. Third, the multiple specialized regressors {f T , f T −1 , · · · , f 1 } are optimized for the resampled distributions of the different stages. This opposes to the single f of (3), which is only optimal for the initial distribution. These differences enable more precise localization than iterative BBox, with no further human engineering.”

**个人感觉，Cascade R-CNN和图（b）最大的不同点在于，图（b）仅仅是用了1个stage，然后多次在同一个RPN中refine得到和gt的IoU更大的proposals，而Cascade个人感觉是用了多个RPN网络，下一级的RPN不止会引入上一级的proposals，而且会进行重采样再次得到同一批anchor，然后一起在新的RPN网络中进行回归？**，而且（b）是在验证阶段采用的方式，而Cascade R-CNN是在训练和验证阶段采用的方式。和（c）的差别也比较明显，Cascade R-CNN中每个stage的输入bbox是前一个stage的bbox输出，而（c）其实没有这种refine的思想，仅仅是检测模型基于不同的IOU阈值训练得到而已。 

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/CascadeRCNN/3.png)



## 三、实验结果

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/CascadeRCNN/4.png)



![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/CascadeRCNN/5.png)



![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/CascadeRCNN/6.png)



![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/CascadeRCNN/7.png)



## 四、结论

* 
* 
