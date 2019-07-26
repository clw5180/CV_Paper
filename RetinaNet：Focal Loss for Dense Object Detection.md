

# **RetinaNet：Focal Loss for Dense Object Detection**

论文地址：<https://arxiv.org/abs/1708.02002>

代码复现：<https://github.com/fizyr/keras-retinanet>

&emsp;&emsp;&emsp;&emsp;&emsp;<https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation> （旋转框）

&emsp;&emsp;&emsp;&emsp;&emsp;<https://github.com/miraclewkf/FocalLoss-MXNet> （优化版的MXNet实现）



## 一、介绍

&emsp;&emsp;RBG和Kaiming大神的新作。我们知道object detection的算法主要可以分为两大类：two-stage detector和one-stage detector。前者是指类似Faster RCNN，RFCN这样需要region proposal的检测算法，这类算法可以达到很高的准确率，但是速度较慢，虽然可以通过减少proposal的数量或降低输入图像的分辨率等方式达到提速，但是速度并没有质的提升；后者是指类似YOLO，SSD这样不需要region proposal，直接回归的检测算法，这类算法速度很快，但是准确率不如前者。作者提出focal loss的出发点也是希望one-stage detector可以达到two-stage detector的准确率，同时不影响原有的速度。

&emsp;&emsp;既然有了出发点，那么就要找one-stage detector的准确率不如two-stage detector的原因。之前观点认为，“单阶段检测器结果不够好的原因是使用的 feature 不够准确（使用一个位置上的 feature），所以需要Roi Pooling这样的 feature aggregation办法得到更准确的表示”。但是这篇文章基本否认了这个观点，**提出one-stage detector不好的原因在于：1、正负样本比例极度不平衡，negative example过多导致loss太大，positive的loss被淹没，不利于收敛  2、gradient被大量easy example支配（虽然easy example的loss很低，但是由于数量众多，所以对loss依然有很大贡献，导致收敛到不够好的一个结果），而且这个才是最核心的因素**。我们知道在object detection领域，一张图像可能生成成千上万的candidate locations，但是其中只有很少一部分是包含object的，这就带来了类别不均衡。那么类别不均衡会带来什么后果呢？引用原文讲的两个后果：(1) training is inefficient as most locations are easy negatives that contribute no useful learning signal; (2) en masse, the easy negatives can overwhelm training and lead to degenerate models. 什么意思呢？负样本数量太大，占总的loss的大部分，而且多是容易分类的，因此使得模型的优化方向并不是我们所希望的那样。其实先前也有一些算法来处理类别不均衡的问题，比如OHEM（online hard example mining），OHEM的主要思想可以用原文的一句话概括：“In OHEM each example is scored by its loss, non-maximum suppression (nms) is then applied, and a minibatch is constructed with the highest-loss examples”。**OHEM算法虽然增加了错分类样本的权重，但是OHEM算法忽略了容易分类的样本**。

&emsp;&emsp;因此针对类别不均衡问题，作者提出一种新的损失函数：**focal loss，这个损失函数是在标准交叉熵损失基础上修改得到的。这个函数可以通过减少易分类样本的权重，使得模型在训练时更专注于难分类的样本**。为了证明focal loss的有效性，作者设计了一个dense detector：RetinaNet，并且在训练时采用focal loss训练。实验证明RetinaNet不仅可以达到one-stage detector的速度，也能有two-stage detector的准确率，coco上AP的提升都在3个点左右，非常显著。
&emsp;&emsp;**focal loss的含义可以看如下Figure1**，横坐标是pt，纵坐标是loss。CE(pt)表示标准的交叉熵公式，FL(pt)表示focal loss中用到的改进的交叉熵，可以看出和原来的交叉熵对比多了一个调制系数（modulating factor）。为什么要加上这个调制系数呢？目的是通过减少易分类样本的权重，从而使得模型在训练时更专注于难分类的样本。首先pt的范围是0到1，所以不管γ是多少，这个调制系数都是大于等于0的。易分类的样本再多，你的权重很小，那么对于total loss的共享也就不会太大。那么怎么控制样本权重呢？举个例子，假设一个二分类，样本x1属于类别1的pt=0.9，样本x2属于类别1的pt=0.6，显然前者更可能是类别1，假设γ=1，那么对于pt=0.9，调制系数则为0.1；对于pt=0.6，调制系数则为0.4，这个调制系数就是这个样本对loss的贡献程度，也就是权重，所以难分的样本（pt=0.6）的权重更大。下图**Figure1中γ=0的蓝色曲线就是标准的交叉熵损失**。<div align="center">

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RetinaNet/1.png)</div>

（自注：个人认为这里pt表示”**是gt的概率，或者和gt的接近程度**“，也就是衡量标准变成了真值，**而不再是之前用p描述的”是1的概率“**；也就是说，如果样本实际值是下面公式中所示的y=1，而p=0.8，则相当于pt和真实值很接近；而如果样本实际值y=0，而p=0.1，同样预测和真实值很接近，那么这里pt=1-0.1=0.9，符合上面的描述。）<div align="center">

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RetinaNet/2.png)</div>

Figure2是在是在COCO数据集上几个模型的实验对比结果。可以看看再AP和time的对比下，本文算法和其他one-stage和two-stage检测算法的差别。<div align="center">

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RetinaNet/3.png)</div>

## 二、主要内容

&emsp;&emsp;看完实验结果和提出算法的出发点，接下来就要介绍focal loss了。在介绍focal loss之前，先来看看交叉熵损失，这里以二分类为例，p表示概率，公式如下：<div align="center">

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RetinaNet/4.png)</div>

因为是二分类，所以y的值是正1或负1，p的范围为0到1。当真实label是1，也就是y=1时，假如某个样本x预测为1这个类的概率p=0.6，那么损失就是-log(0.6)，注意这个损失是大于等于0的。如果p=0.9，那么损失就是-log(0.9)，所以p=0.6的损失要大于p=0.9的损失，这很容易理解。

为了方便，用pt代替p，如下公式（2）。这里的pt就是最前面图1中的横坐标。<div align="center">

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RetinaNet/5.png)</div>

接下来介绍一个最基本的对交叉熵的改进，也将作为本文实验的baseline，如下公式3。什么意思呢？增加了一个系数at，跟pt的定义类似，当label=1的时候，at=a；当label=-1的时候，at=1-a，a的范围也是0到1。**因此可以通过设定a的值（一般而言假如1这个类的样本数比-1这个类的样本数多很多，那么a会取0到0.5来增加-1这个类的样本的权重）来控制正负样本对总的loss的共享权重**。<div align="center">

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RetinaNet/6.png)</div>

**显然前面的公式3虽然可以控制正负样本的权重，但是没法控制容易分类和难分类样本的权重**，于是就有了focal loss：<div align="center">

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RetinaNet/7.png)

</div>

这里的γ称作focusing parameter，γ>=0。作者在这里给了一个直观的例子：“**For instance, with γ = 2, an**
**example classified with pt = 0:9 would have 100× lower loss compared with CE and with pt ≈ 0:968 it would have 1000× lower loss**. This in turn increases the importance of correcting misclassified examples (whose loss is scaled down by at most 4× for pt ≤ 0.5 and γ = 2) ；

<div align="center">

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RetinaNet/8.png)

</div>

称为调制系数（modulating factor） 。
**focal loss的两个重要性质：当一个样本被分错的时候，pt是很小的（比如当y=1时，p一般小于0.5被认为是错分类，此时pt就比较小），因此调制系数就趋于1，也就是说相比原来的loss是没有什么大的改变的。当pt趋于1的时候，此时分类正确而且是易分类样本，调制系数趋于0，也就是对于总的loss的贡献很小。** 

**作者在实验中采用的是公式5的focal loss（结合了公式（3）和公式（4），这样既能调整正负样本的权重，又能控制难易分类样本的权重）：**<div align="center">

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RetinaNet/9.png)</div>

在实验中a的选择范围也很广，一般而言当γ增加的时候，a需要减小一点（实验中**γ=2，a=0.25的效果最好**）



**RetinaNet的结构图**如下：<div align="center">

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RetinaNet/10.png)</div>

**Feature Pyramid Network Backbone：**
**FPN** 作为 Backbone。它在ResNet网络上**增加了top-down(自顶向下)通路和lateral(侧向连接)通路**，从图片的单一分辨率构建丰富的、多尺度的特征金字塔。**金字塔的每一层特征用来检测不同尺寸的目标**。
本文FPN**由P3-P7构成**（Pn层的分辨率和输入图像相比缩小2^n倍），P3-P5是由ResNet的C3-C5计算，P6是由C5使用stride=2的3X3卷积得到，P7是由P6经过stride=2的3x3卷积得到，特征金字塔所有层的Channel=256。与原始的FPN不同之处在于：（1）**这里没有使用P2层，作者说是for computional reasons**；（2）**P6是由stride=2的卷积得到不是降采样**；（3）**引入P7层提升对大尺寸目标的检测效果**。
需要强调的是使用FPN作为主干网的原因是，实验发现如果只使用ResNet层，最终mAP值较低。

**Anchors：**
类似RPN具有平移不变性的anchor boxes。从P3到P7层的anchors的面积从32x32依次增加到512x512。每层anchors ratios（长宽比）包括{1:2, 1:1, 2:1}。每层scales包括{2^0, 2^1/3, 2^2/3}；这样每层有9个anchors，通过不同层覆盖了输入图像 32~813 像素区间。
每个Anchor会有长度为K(class)的one-hot分类目标和4-vector的box回归目标。这里作者仿照RPN的做法，但是做了一些修改：
(1) 对于anchor是否与GT关联，依然根据二者IOU的阈值，这里设置为0.5（RPN是0.7），即如果IOU大于0.5，则anchors和GT关联；IOU在[0, 0.4)作为背景。
(2) 每个anchor最多关联一个GT；K(class)的one-hot中关联的类别为1，其它为0。
(3) 边框回归就是计算anchor到关联的GT之间的偏移。

**Classification Subnet：**
 连接在FPN每层的FCN，参数共享。Feature Map，使用4个3×3的卷积层，每个卷积层接一个ReLU层，然后是channel=KA(K是类别数，A是anchor数)的3×3卷积层，最后使用sigmoid激活函数。
与RPN相比，网络更深，只使用了3×3卷积；不和边框回归子网络共享参数。

**Box Regression Subnet：**
 结构同上，最后一层channel=4A。

**Inference and Training：**
**Inference：**为了提高速度，只对FPN每层部分predictions处理。FPN的每个特征层，首先使用0.05的阈值筛选出是前景的object，最多选取前1k个predictions进行后续处理。融合各层的predictions，再使用NMS（阈值0.5）去掉重叠box。



## 三、实验结果

&emsp;&emsp;Table1是关于RetinaNet和Focal Loss的一些实验结果。（a）是在交叉熵的基础上加上参数a，a=0.5就表示传统的交叉熵，可以看出当a=0.75的时候效果最好，AP值提升了0.9。（b）是对比不同的参数γ和a的实验结果，可以看出随着γ的增加，AP提升比较明显。（d）通过和OHEM的对比可以看出最好的Focal Loss比最好的OHEM提高了3.2AP。这里OHEM1:3表示在通过OHEM得到的minibatch上强制positive和negative样本的比例为1:3，通过对比可以看出这种强制的操作并没有提升AP。（e）加入了运算时间的对比，可以和前面的Figure2结合起来看，速度方面也有优势！注意这里RetinaNet-101-800的AP是37.8，当把训练时间扩大1.5倍同时采用scale jitter，AP可以提高到39.1，这就是全文和table2中的最高的39.1AP的由来。<div align="center">

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RetinaNet/11.png)</div>

Figure4是对比forground和background样本在不同γ情况下的累积误差。纵坐标是归一化后的损失，横坐标是总的foreground或background样本数的百分比。可以看出γ的变化对正（forground）样本的累积误差的影响并不大，但是对于负（background）样本的累积误差的影响还是很大的（γ=2时，将近99%的background样本的损失都非常小）。<div align="center">

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RetinaNet/12.png)</div>



## 四、结论

&emsp;&emsp;原文的这段话概括得很好：**In this work, we identify class imbalance as the primary obstacle preventing one-stage object detectors from surpassing top-performing, two-stage methods, such as Faster R-CNN variants. To address this, we propose the focal loss which applies a modulating term to the cross entropy loss in order to focus learning on hard examples and down-weight the numerous easy negatives.**



参考：https://blog.csdn.net/u014380165/article/details/77019084 

&emsp;&emsp;&emsp;https://www.zhihu.com/people/naiyan-wang/answers/by_votes