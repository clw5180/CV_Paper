

# **YOLOv2（YOLO9000）：Better，Faster，Stronger**

论文地址：<<https://arxiv.org/abs/1612.08242>>

代码复现：<https://github.com/longcw/yolo2-pytorch> （PyTorch版本）



## 一、介绍

&emsp;&emsp;**在保持原有速度的优势之下，精度上得以提升**。VOC 2007数据集测试，67FPS下mAP达到76.8%，准确率比V1高出20%，可以与Faster R-CNN和SSD一战。



## 二、YOLOv2精度的改进（Better）

YOLOv2中使用的技巧，以及提升的mAP：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/1.png)



### Batch Normalization

在V1的基础上增加了BN层，位置在Conv层的后面。BN的两个作用：

1、首先，**BN可以比较好地解决过拟合的问题，因此不再使用dropout**（自注：BN算法时如何防止过拟合的？在这里摘录一段国外大神的解释：“When training with Batch Normalization, a training example is seen in conjunction with other examples in the mini-batch, and the training network no longer producing deterministic values for a given training example. In our experiments, we found this effect to be advantageous to the generalization of the network.”大概意思是：**在训练中，BN的使用使得一个mini-batch中的所有样本都被关联在了一起，因此网络不会从某一个训练样本中生成确定的结果**。意思就是同样一个样本的输出不再仅仅取决于样本本身，也取决于跟这个样本属于同一个mini-batch的其它样本。同一个样本跟不同的样本组成一个mini-batch，它们的输出是不同的（仅限于训练阶段，在inference阶段是没有这种情况的）。我把这个理解成一种数据增强：同样一个样本在超平面上被拉扯，每次拉扯的方向的大小均有不同。不同于数据增强的是，这种拉扯是贯穿数据流过神经网络的整个过程的，意味着神经网络每一层的输入都被数据增强处理了。**相比于Dropout、L1、L2正则化来说，BN算法防止过拟合效果并不是很明显**）。

2、其次，BN的主要作用在于：对于每个隐层神经元，**把逐渐向**非线性函数映射后向取值区间极限**饱和区靠拢的输入分布强制拉回到均值为0方差为1的比较标准的正态分布**，使得非线性变换函数的输入值落入对输入比较敏感的区域，以此**解决了梯度消失问题**。YOLO网络在每一个卷积层后添加batch normalization，通过这一方法，mAP获得了2个点的提升。（这一操作在yolo_v3上依然有所保留，BN层从v2开始便成了yolo算法的标配）

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/2.jpg)

https://www.cnblogs.com/guoyaohua/p/8724433.html



### High Resolution Classifier

**目前的目标检测方法中**，基本上都会使用ImageNet预训练过的模型（classifier）来提取特征；此前的 yolo v1 在预训练时采用的是 224 x 224 的图像输入，这是为了与AlexNet等等预训练网络的输入一致，然后再**检测的时候采用 448 x 448 的输入**。这样做的一个问题在于从训练模型切换到检测模型时，模型需要对输入大小做出调整，这在一定程度上影响了 YOLOv1 的准确性。因此在YOLOv2中，作者首先对分类网络（自定义的darknet）进行了fine tune，**预训练时先在 224 x 224 的输入上训练 160 轮，然后再调整输入大小至 448 x 448再训练 10 轮**，此时训练后的网络就可以适应高分辨率的输入了。实验表明通过这样的改进使得 **mAP 提升了 4个点**。



### Convolutional With Anchor Boxes

之前的YOLO利用全连接层的数据完成边框的预测，导致丢失较多的空间信息，定位不准。作者在这一版本中借鉴了Faster R-CNN中的anchor思想，回顾一下，anchor是RPN网络中的一个关键步骤，说的是在卷积特征图上进行滑窗操作，每一个中心可以预测9种不同大小的建议框。看到YOLOv2的这一借鉴，我只能说SSD的作者是有先见之明的。为了引入anchor boxes来预测bounding boxes，作者在网络中果断去掉了全连接层。剩下的具体怎么操作呢？**首先，作者去掉了后面的一个池化层以确保输出的卷积特征图有更高的分辨率**。然后，通过**缩减网络，让图片输入分辨率为416 x 416**，这一步的目的是为了让后面产生的卷积**特征图宽高都为奇数，这样就可以产生一个center cell**。**作者观察到，大物体通常占据了图像的中间位置， 就可以只用中心的一个cell来预测这些物体的位置，否则就要用中间的4个cell来进行预测**，这个技巧可以稍稍提升效率。最后，YOLOv2使用了卷积层降采样（factor为32），使得输入卷积网络的416 x 416图片最终得到**13 x 13**的卷积特征图（416 / 32=13）。加入了anchor boxes后，可以预料到的结果是**召回率上升，准确率下降**。我们来计算一下，假设每个cell预测9个建议框，那么**总共会预测13 * 13 * 9 = 1521个boxes，而之前的网络仅仅预测7 * 7 * 2 = 98个boxes**。具体数据为：没有anchor boxes，模型recall为81%，mAP为69.5%；加入anchor boxes，模型recall为88%，mAP为69.2%。这样看来，准确率只有小幅度的下降，而召回率则提升了7%，说明可以通过进一步的工作来加强准确率，的确有改进空间。



### Dimension Clusters（维度聚类）

作者在使用anchor的时候遇到了两个问题，**第一个是anchor boxes的宽高维度往往是精选的先验框（hand-picked priors）**，虽说在训练过程中网络也会学习调整boxes的宽高维度，最终得到准确的bounding boxes。但是，如果一开始就选择了更好的、更有代表性的先验boxes维度，那么网络就更容易学到准确的预测位置。


和以前的精选boxes维度不同，作者使用了K-means聚类方法类训练bounding boxes，可以自动找到更好的boxes宽高维度。传统的方法使用的是欧氏距离函数，也就意味着较大的boxes会比较小的boxes产生更多的error，聚类结果可能会偏离。为此，作者采用的评判标准是IOU得分（也就是boxes之间的交集除以并集），这样的话，error就和box的尺度无关了，最终的距离函数为：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/3.png)

作者通过改进的**K-means对训练集中的boxes进行了聚类，判别标准是平均IoU得分**，聚类结果如下图：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/4.png)

可以看到，平衡复杂度和IOU之后，最终得到k值为5，意味着作者选择了5种大小的box维度来进行定位预测，这与手动选择的box维度不同。结果中**扁长的框较少，而瘦高的框更多（这符合行人的特征）**，这种结论如不通过聚类实验恐怕是发现不了的。作者也做了实验来对比两种策略的优劣，如下图，使用聚类方法，仅仅5种boxes的召回率就和Faster R-CNN的9种相当。说明K-means方法的引入使得生成的boxes更具有代表性，为后面的检测任务提供了便利。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/5.png)



### Direct location prediction（直接位置预测）

那么，作者在使用anchor boxes时发现的**第二个问题就是：模型不稳定，尤其是在早期迭代的时候**。大部分的不稳定现象出现在预测box的  使用的是如下公式：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/6.png)

**注：可能这个公式有误，作者应该是把加号写成了减号**。理由如下，anchor的预测公式来自于Faster-RCNN，我们来看看人家是怎么写的：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/7.png)

公式中，符号的含义解释一下：x 是坐标预测值，xa是anchor坐标（预设固定值），x∗是坐标真实值（标注信息），其他变量 y，w，h 以此类推，t 变量是偏移量。然后把前两个公式变形，就可以得到正确的公式。**这个公式的理解为：当预测 tx=1，就会把box向右边移动一定距离（具体为anchor box的宽度），预测 tx=−1，就会把box向左边移动相同的距离**。


这个公式没有任何限制，使得无论在什么位置进行预测，任何anchor boxes可以在图像中任意一点结束（我的理解是， 没有数值限定，可能会出现anchor检测很远的目标box的情况，效率比较低。正确做法应该是每一个anchor只负责检测周围正负一个单位以内的目标box）。模型随机初始化后，需要花很长一段时间才能稳定预测敏感的物体位置。

在此，作者就没有采用预测直接的offset的方法，而使用了预测相对于grid cell的坐标位置的办法，作者又把ground truth限制在了0到1之间，利用logistic回归函数来进行这一限制。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/8.png)

现在，神经网络在特征图（13 *13 ）的每个cell上预测5个bounding boxes（聚类得出的值），同时每一个bounding box预测5个值，那么预测值可以表示为：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/9.png)

这几个公式参考上面Faster-RCNN和YOLOv1的公式以及下图就比较容易理解。**tx, ty经sigmod函数处理过，取值限定在了0~1，实际意义就是使anchor只负责周围的box，有利于提升效率和网络收敛**。σ 函数的意义没有给，但估计是把归一化值转化为图中真实值，使用 e 的幂函数是因为前面做了 ln 计算，因此，σ(tx)是bounding box的中心相对栅格左上角的横坐标，σ(ty)是纵坐标，σ(to)是bounding box的confidence score。


定位预测值被归一化后，参数就更容易得到学习，模型就更稳定。作者使用Dimension Clusters和Direct location prediction这两项anchor boxes改进方法，mAP获得了5%的提升。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/10.png)



### Fine-Grained Features（细粒度特征）

上述网络上的修改使YOLO最终在13 * 13的特征图上进行预测，虽然这足以胜任大尺度物体的检测，但是用上细粒度特征的话，这可能对小尺度的物体检测有帮助。Faser R-CNN和SSD都在不同层次的特征图上产生区域建议（SSD直接就可看得出来这一点），获得了多尺度的适应性。这里使用了一种不同的方法，简单添加了一个转移层（ passthrough layer），这一层要把浅层特征图（分辨率为26 * 26，是底层分辨率4倍）连接到深层特征图。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/11.jpg)

这个转移层也就是把高低两种分辨率的特征图做了一次连结，连接方式是叠加特征到不同的通道而不是空间位置，类似于Resnet中的identity mappings。这个方法把26 * 26 * 512的特征图连接到了13 * 13 * 2048的特征图，这个特征图与原来的特征相连接。YOLO的检测器使用的就是经过扩张的特征图，它可以拥有更好的细粒度特征，使得模型的性能获得了1%的提升。（这段理解的也不是很好，要看到网络结构图才能清楚）

补充：关于passthrough layer，具体来说就是特征重排（不涉及到参数学习），前面26 * 26 * 512的特征图使用按行和按列隔行采样的方法，就可以得到4个新的特征图，维度都是13 * 13 * 512，然后做concat操作，得到13 * 13 * 2048的特征图，将其拼接到后面的层，相当于做了一次特征融合，有利于检测小目标。



### Multi-Scale Training

原来的YOLO网络使用固定的448 * 448的图片作为输入，现在加入anchor boxes后，输入变成了416 * 416。目前的网络只用到了卷积层和池化层，那么就可以进行动态调整（意思是可检测任意大小图片）。作者希望YOLOv2具有不同尺寸图片的鲁棒性，因此在训练的时候也考虑了这一点。

不同于固定输入网络的图片尺寸的方法，作者在几次迭代后就会微调网络。没经过10次训练（10 epoch），就会随机选择新的图片尺寸。YOLO网络使用的降采样参数为32，那么就使用32的倍数进行尺度池化{320,352，…，608}。最终最小的尺寸为320 * 320，最大的尺寸为608 * 608。接着按照输入尺寸调整网络进行训练。

这种机制使得网络可以更好地预测不同尺寸的图片，意味着同一个网络可以进行不同分辨率的检测任务，在小尺寸图片上YOLOv2运行更快，在速度和精度上达到了平衡。

&emsp;&emsp;在小尺寸图片检测中，YOLOv2成绩很好，输入为228 * 228的时候，帧率达到90FPS，mAP几乎和Faster R-CNN的水准相同。使得其在低性能GPU、高帧率视频、多路视频场景中更加适用。在大尺寸图片检测中，YOLOv2达到了先进水平，VOC2007 上mAP为78.6%，仍然高于平均水准，下图是YOLOv2和其他网络的成绩对比：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/12.png)

作者在VOC2012上对YOLOv2进行训练，下图是和其他方法的对比。YOLOv2精度达到了73.4%，并且速度更快。同时YOLOV2也在COCO上做了测试（IOU=0.5），也和Faster R-CNN、SSD作了成绩对比。总的来说，比上不足，比下有余。



## 三、YOLOv2速度的改进（Faster）

YOLO一向是速度和精度并重，作者为了改善检测速度，也作了一些相关工作。

大多数检测网络有赖于VGG-16作为特征提取部分，VGG-16的确是一个强大而准确的分类网络，但是复杂度有些冗余。224 * 224的图片进行一次前向传播，其卷积层就需要多达306.9亿次浮点数运算。

YOLOv2使用的是基于Googlenet的定制网络，比VGG-16更快，一次前向传播仅需85.2亿次运算。可是它的精度要略低于VGG-16，单张224 * 224取前五个预测概率的对比成绩为88%和90%（低一点点也是可以接受的）。

### Darknet-19

YOLOv2使用了一个新的分类网络作为特征提取部分，参考了前人的先进经验，比如类似于VGG，作者使用了较多的3 * 3卷积核，在每一次池化操作后把通道数翻倍。借鉴了network in network的思想，网络使用了全局平均池化（global average pooling），把1 * 1的卷积核置于3 * 3的卷积核之间，用来压缩特征。也用了batch normalization（前面介绍过）稳定模型训练。 

最终得出的基础模型就是Darknet-19，如下图，其包含19个卷积层、5个最大值池化层（maxpooling layers ），下图展示网络具体结构。Darknet-19运算次数为55.8亿次，imagenet图片分类top-1准确率72.9%，top-5准确率91.2%。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv2/13.png)

Training for classification

作者使用Darknet-19在标准1000类的ImageNet上训练了160次，用的随机梯度下降法，starting learning rate 为0.1，polynomial rate decay 为4，weight decay为0.0005 ，momentum 为0.9。训练的时候仍然使用了很多常见的数据扩充方法（data augmentation），包括random crops, rotations, and hue, saturation, and exposure shifts。 （这些训练参数是基于darknet框架，和caffe不尽相同）

初始的224 * 224训练后，作者把分辨率上调到了448 * 448，然后又训练了10次，学习率调整到了0.001。高分辨率下训练的分类网络在top-1准确率76.5%，top-5准确率93.3%。

Training for detection

分类网络训练完后，就该训练检测网络了，作者去掉了原网络最后一个卷积层，转而增加了三个3 * 3 * 1024的卷积层（可参考darknet中cfg文件），并且在每一个上述卷积层后面跟一个1 * 1的卷积层，输出维度是检测所需的数量。对于VOC数据集，预测5种boxes大小，每个box包含5个坐标值和20个类别，所以总共是5 * （5+20）= 125个输出维度。同时也添加了转移层（passthrough layer ），从最后那个3 * 3 * 512的卷积层连到倒数第二层，使模型有了细粒度特征。

作者的检测模型以0.001的初始学习率训练了160次，在60次和90次的时候，学习率减为原来的十分之一。其他的方面，weight decay为0.0005，momentum为0.9，依然使用了类似于Faster-RCNN和SSD的数据扩充（data augmentation）策略。



## 四、YOLOv2分类的改进（Stronger）
这一部分，作者使用联合训练方法，结合wordtree等方法，使YOLOv2的检测种类扩充到了上千种，具体内容待续。



## 五、总结和展望

* 作者大概说的是，之前的技术改进对检测任务很有帮助，在以后的工作中，可能会涉足弱监督方法用于图像分割。监督学习对于标记数据的要求很高，未来要考虑弱标记的技术，这将会极大扩充数据集，提升训练量。




参考：https://blog.csdn.net/jesse_mx/article/details/53925356