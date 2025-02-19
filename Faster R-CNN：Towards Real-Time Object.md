论文：Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
============

论文地址：https://arxiv.org/abs/1506.01497

代码复现：https://github.com/smallcorgi/Faster-RCNN_TF

​                   https://github.com/endernewton/tf-faster-rcnn

​                   https://github.com/rbgirshick/py-faster-rcnn

​                   https://github.com/chenyuntc/simple-faster-rcnn-pytorch（pytorch极简实现，by陈云） 

​                   https://github.com/liuyuemaicha/simple_faster_rcnn（pytorch极简实现带注释）



## 一、介绍

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/1.png)

&emsp;&emsp;Faster R-CNN，可以大致分为两个部分，一个是**RPN（区域生成网络）**，另一个是**Fast R-CNN网络**，前者是一种候选框（proposal）的推荐算法，而后者则是在此基础上对框的位置和框内的物体的类别进行细致计算。（可以将Faster R-CNN可以简单地看做“RPN + Fast RCNN“的系统，用区域生成网络代替Fast R-CNN中的Selective Search方法）



## 二、主要内容

&emsp;&emsp;Faster RCNN主要可以分为如下**四个部分**：      

（1）首先向CNN网络（**ZF**、**VGGNet-16**或**ResNet**等）输入任意大小图片（自注：论文中短边压缩到600像素，长边不确定，论文中以1000为例进行分析）；VGG16的结构如下图所示：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/2_0.png)

Conv layers部分共有13个conv层，13个relu层，4个pooling层。注意在Faster RCNN Conv layers中对所有的卷积都做了扩边处理（ **pad=1**，即填充一圈0），导致原图变为 (M+2)x(N+2)大小，再做3x3卷积后输出MxN 。正是这种设置，导致Conv layers中的conv层不改变输入和输出矩阵大小。类似的是，Conv layers中的pooling层kernel_size=2，stride=2。这样每个经过pooling层的MxN矩阵，都会变为(M/2)x(N/2)大小。综上所述，在整个Conv layers中，conv和relu层不改变输入输出大小，只有pooling层使输出长宽都变为输入的1/2。那么，一个MxN大小的矩阵经过Conv layers固定变为(M/16) x (N/16)，这样Conv layers生成的feature map中都可以和原图对应起来。

（2）将**conv5_3**输出的feature maps送入RPN，用于生成region proposals。得到**~20k个anchors**（假设图片尺寸约为**1000x600，经过前级VGGNet-16图片的宽高各缩小1/16，则产生62x37x9个anchors**）的坐标和类别；如果是训练阶段，还会去掉所有超出边界之外的anchor，剩下**~6k个anchors**；之后采用**非极大值抑制**（论文中NMS的IoU阈值为0.7），每张图片得到**~2k个region proposals**，输出得分**Top-N**的region proposals；该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的region proposals（或称为 regions of interest，RoI）。RPN自己选出了256个anchor进行训练，然后将筛选出的~2000个RoIs送入RoI Pooling层。

（3）RoI Pooling层收集输入的feature maps和proposals，综合这些信息后提取**固定大小**的**proposal feature maps**，送入fc层进行分类和回归。（**自注：对于ResNet，貌似是在conv4层提取feature maps，然后得到proposal feature maps送入conv5做全卷积，相当于fc层**，代码如下：）

```python
def restnet_head(input, is_training, scope_name): 
    block4 = [resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5, _ = resnet_v1.resnet_v1(input,
                                    block4,
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)
        # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')
        C5_flatten = tf.reduce_mean(C5, axis=[1, 2], keep_dims=False, name='global_average_pooling')
        # C5_flatten = tf.Print(C5_flatten, [tf.shape(C5_flatten)], summarize=10, message='C5_flatten_shape')

    # global average pooling C5 to obtain fc layers
    return C5_flatten
```

（4）最后将**proposal feature maps**送入后续fc层，然后一路经过softmax网络作classification，另一路再做一次bounding box regression获得检测框最终的精确位置。

（自注：以上主要参考：<https://zhuanlan.zhihu.com/p/31426458>）

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/2_1.png)



&emsp;&emsp;各conv层的维度以及各层输入、输出的维度图如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/3.jpg)



### Fast RCNN详解

#### 1、backbone部分

1、特征提取（自注：主要参考 <https://www.cnblogs.com/wangyong/p/8513563.html>）

对与每张图片，首先需要进行如下数据处理：

- 图片进行缩放，使得长边小于等于1000，短边小于等于600（至少有一个等于）。
- 对相应的bounding boxes 也也进行同等尺度的缩放。
- 对于Caffe 的VGG16 预训练模型，需要图片位于0-255，BGR格式，并减去一个均值，使得图片像素的均值为0。

最后返回四个值供模型训练：

- images ： 3×H×W ，BGR三通道，宽W，高H
- bboxes： 4×K , K个bounding boxes，每个bounding box的左上角和右下角的座标，形如（Y_min,X_min, Y_max,X_max）,第Y行，第X列。
- labels：K， 对应K个bounding boxes的label（对于VOC取值范围为[0-19]）
- scale: 缩放的倍数, 原图H' ×W'被resize到了HxW（scale=H/H' ）

**需要注意的是，目前大多数Faster R-CNN实现都只支持batch-size=1的训练**（[这个](https://zhuanlan.zhihu.com/github.com/jwyang/faster-rcnn.pytorch) 和[这个](https://link.zhihu.com/?target=https%3A//github.com/precedenceguo/mx-rcnn)实现支持batch_size>1）。

我们可以假定图片尺寸为=1000 x 600（如果图片少于该尺寸，可以边缘补0，即图像会有黑色边缘）

①   13个conv层：kernel_size=3, pad=1, stride=1;

&emsp;&emsp;卷积公式：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/4.png)

&emsp;&emsp;所以，conv层不会改变图片大小（即：输入的图片大小=输出的图片大小）

②   13个relu层：激活函数，不改变图片大小

③   4个pooling层：kernel_size=2, stride=2; pooling层会让输出图片是输入图片的1/2

&emsp;&emsp;经过Conv layers，图片大小变成 (M / 16) x (N / 16) ，即：62 x 37 (1000 / 16 = 62.5, 600 / 16 = 37.5)；则，Feature Map就是 **62 x 37 x 512维** （**注：VGG16的conv5_3输出的通道数量是512-d，而ZF是256-d，后面都以VGG为例**），表示Feature Map的大小为62 x 37，通道数为512；

&emsp;&emsp;另外注意，为了节省显存，一般设置conv1和conv2的学习率设为0。



#### 2、RPN部分

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/5.png)

&emsp;&emsp;从上面的图可以看到，RPN网络实际分为2条线，上面一条通过softmax对anchors进行分类，获得positive和negative分类，下面一条用于计算对于anchors的bounding box regression偏移量，以获得精确的proposal。而最后的Proposal层则负责 **综合** positive anchors 和对应bounding box regression偏移量获取proposals，同时**剔除超出边界的proposals**。其实整个网络到了Proposal Layer这里，就完成了相当于目标定位的功能。

&emsp;&emsp;为了进一步更清楚的看懂RPN的工作原理，将Caffe版本下的网络图贴出来，对照网络图进行讲解会更清楚。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/6.png)

&emsp;&emsp在Feature Map进入RPN后，先经过rpn_conv/3x3，进行3x3的卷积，应该是使用了padding，能够保证输出大小依然是62 x 37，通道数量是512，这样做的目的应该是进一步集中特征信息。接着看到两个全卷积，分别是图中的‘rpn_cls_score’和‘rpn_bbox_pred’，kernel_size=1x1，p=0，stride=1；

&emsp;&emsp;在Feature Map进入RPN后，先经过rpn_conv/3x3，进行3x3的卷积，应该是使用了padding，能够保证输出大小依然是**62 x 37，通道数量是512**，这样做的目的应该是**进一步集中特征信息**。接着看到两个全卷积，分别是图中的‘rpn_cls_score’和‘rpn_bbox_pred’，kernel_size=1x1，p=0，stride=1；

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/7.png)

①   **rpn_cls**：62 x 37 x 512-d  ⊕  1x1x512x18 ==>  **62 x 37 x 9 x 2** 

&emsp;&emsp;逐像素对其9个Anchor box进行二分类

②   **rpn_bbox**：62 x 37 x 512-d  ⊕  1x1x512x36 ==> **62 x 37 x 9 x 4**

&emsp;&emsp;逐像素得到其9个Anchor box四个坐标信息（其实是偏移量，后面介绍）

  如下图所示：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/8.png)

&emsp;&emsp;（对于上面特征图中**62 x 37个"位置"（自注：这里不用乘以256），每一个"位置"都"负责”原图中对应位置的9种尺寸框的检测**，检测的目标是判断框中是否存在一个物体，因此**共有62x37x9个“框”**。在Faster R-CNN原论文中，将这些框都统一称为"**anchor**"）。



#### 2.2.1）Anchor

&emsp;&emsp;前面提到经过Conv layers后，feature map大小变成了原图的1/16，故stride=16（stride表示 sliding windows在feature map上滑过1个像素，相当于在原图上滑动16个像素），在生成Anchors时，我们先定义一个base_anchor，大小为16x16的box（因为62x37的特征图上的一个点，可以对应到原图（1000x600）上一个16x16大小的区域），源码中转化为**[0, 0, 15, 15]**的数组（TODO：这个是什么意思？）。论文中用到的anchor有3种尺寸和3种比例，参数ratios=[0.5, 1, 2]  scales=[128, 256, 512] （自注：这里的scales是映射到原图上的尺寸，而不是feature map上的尺寸）；**另外scales代表的是正方形anchor的边长**而不是面积）。对这3个scale分别产生长、宽比分别为[0.5, 1, 2]的anchor box，如下图所示（**自注：实际上对于feature map的1个pixel，应该产生中心点在同一位置的9个anchor，而这里画成中心点不在同一位置，应该只是为了更直观地表达**）。**比如对于ratio=2的anchor，并不是说w=256，h=128，而是保证面积和 128^2 相同，来求宽和高**，即解这两个方程：w x h=128^2 ， w / h =2，在代码中可以直接写成 **h = 128 / sqrt(ratio)** ，相当于h变为原来的1 / 1.414，而w变为原来的1.414倍，求出h=90.5，w=181。综上，根据3个scales和3个ratios，生成9个Anchor box。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/12.jpg)

&emsp;&emsp;另外，论文中提到不同anchor学习到的proposal的平均尺寸如下表，这个值是怎么来的？为什么同样是anchor_scale = 128，ratio=2:1和ratio=1:2差这么多？（对于188x111这个值的解释：In `./lib/rpn/generate_anchors.py`, you can find that the anchor `128x128` with ration `2:1` is `[-83 -39 100 56]` whose size is `184x96`. And the author calculated an average size of the proposals. So it will have some differences with the actual sizes.......另外这位网友提到原图resize的问题，it has a minimum dimension(height/width) of 600 pixels and a max dimension(width/height) of 1024 pixels. (Although I do not check the code in Tensorflow version, I am not very sure.) For example, if you have an image of size 400x512, it will be resized to 600x768; if the original image size is 100x512, the code will resize it to 200x1024. Something like this....  详见<https://github.com/rbgirshick/py-faster-rcnn/issues/112>）

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/13.png)

&emsp;&emsp;

&emsp;&emsp;知乎有人提到：“**anchor的本质类似于SPP (spatial pyramid pooling) 思想的逆向**。而SPP就是将不同尺寸的输入resize成为相同尺寸的输出。所以SPP的逆向就是，**将相同尺寸的输出，倒推得到不同尺寸的输入**。” <https://www.zhihu.com/question/42205480> ，

。。。

&emsp;&emsp;base_anchor = [0,0,15,15] 生成的9个Anchor box坐标如下（自注：个人的理解是，取一个左上角在(0, 0)位置，右下角在(15, 15)，中心位于(7.5, 7.5)位置的anchor为基础，计算映射到原图的9个anchor的左下角和右上角坐标（**自注：下面生成9个anchor坐标可以看到，虽然面积已经增大到128^2 ~ 512^2，但中心位置还是(7.5, 7.5)，即在feature map上，暂时并没有映射到原图，之后再进行处理**）。个人感觉主要就是为了计算出对应到原图的9个anchor的**偏移量**，这样有了一个基准之后，就无须重复计算其他位置的anchor坐标，直接看距离这个base_anchor多少，然后加上相应的偏移量就可以了。实际中，比如一幅1000x600的图，得到feature map尺寸为62x37，首先经过3x3的卷积，既然输出尺寸保持不变，必然是经过了padding，那么中心点的横纵坐标最开始应该在(0.5, 0.5)的位置，即最靠左上角的小方块的中心坐标，以此为中心进行第一次卷积。然后滑动窗移动1个像素之后中心位置变成(1.5, 0.5).......）：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/14.png)

（以下结合自己pycharm调试，并参考 https://zhuanlan.zhihu.com/p/35922980，推导一下这几个坐标是怎么来的）

&emsp;&emsp;首先，上面每一行四个数分别代表xmin, ymin, xmax, yamx。中心点坐标都是(7.5, 7.5)。分别对应了三个box面积尺寸分别为[128, 256, 512]，进行宽高ratio=[0.5, 1, 2]三个尺寸的缩放。过程如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/14_0.png)

- #### anchor的值是由如下步骤计算得来的:

首先设置基础anchor，默认为[0, 0, 15, 15]。

（1）计算基础anchor的宽为16，高为16，anchor中心(7.5,7.5)，以及面积256；
（2）这里我理解是，源码中w = sqrt(scale / ratio)，相当于上面提到的解方程的过程，不过这里显然ratio的含义变成了高宽比 h / w，所以上面的式子先求出w ，得到w = [23, 16, 11]
（3）anchor的高度h由[23, 16, 11] * [0.5, 1, 2]确定,得到[12, 16, 22]，也就是说，在scale=16也就是面积为256的feature map上，
（5）由anchor的中心以及不同的宽和高可以得到此时的anchors.即

```
array([[ -3.5,   2. ,  18.5,  13. ],
       [  0. ,   0. ,  15. ,  15. ],
       [  2.5,  -3. ,  12.5,  18. ]])
```

（6）再利用三种不同的scales[8,16,32]分别去扩大anchors，扩大的方法是先计算出来上一步的anchor的中心以及宽高，使宽高分别乘以scale，然后再利用中心和新的宽高计算出最终所要的anchors。这样得到对每一行处理，第一行[-3.5,2,18.5,13]得到[23.0,12.0,7.5,7.5]。对宽和高取 [8, 16, 32] 三个比例值得到w:[ 184,368,736.], h:[96,192, 384]。这样然后再以(7.5, 7.5)为中心得到3个anchor。比如w=184, h=96。得到：-84 -40 99 55这个anchor；其他anchor同理。

```
[[ -84.  -40.   99.   55.]
 [-176.  -88.  191.  103.]
 [-360. -184.  375.  199.]
 [ -56.  -56.   71.   71.]
 [-120. -120.  135.  135.]
 [-248. -248.  263.  263.]
 [ -36.  -80.   51.   95.]
 [ -80. -168.   95.  183.]
 [-168. -344.  183.  359.]]
```

anchor得到以后都要映射到原图像上的。上面说了一个点有9个anchor，那么对于特征图大小[P, Q] 每个点都有9个anchor。但是[P, Q, 512]不是原图像，我们要找到[P, Q]上每个点和原图像[W, H]每个点的对应关系。方便说明我们假设P =4, Q=4。那么P,Q在原图像对应点左边就是：

P/Q:[0, 1, 2, 3]      原图上W/H：[0, 16, 32, 48]

因为一般图像上原点是图像左上角，向下为y轴正方向（高），向右为x正方向（宽）。分别以W为图像横坐标位置，H为纵坐标得到(0, 0),(0, 16), (0, 32), (0, 48), (16, 0)， (16, 16), ….(48, 48)。一共16个坐标点。然后以每个坐标点为中心在应用上面9个anchor，每个点得到9个box。每个新的anchor=[xmin + x_ctre, ymin+y_ctre, xmax+x_ctre, ymax+y_ctre]。这样就得到了初始的每个点的anchor。

这样就得到了初始的每个点的anchor。我们用 ![[公式]](https://www.zhihu.com/equation?tex=%5Bx_%7Ba%7D%2Cy_%7Ba%7D%2Cw_%7Ba%7D%2Ch_%7Ba%7D%5D) 来表示上面得到的每个anchor的中心点坐标以及这个anchor的宽和高，我们称其为标准的anchor输出。其实这里的anchor已经可以叫box了。

&emsp;&emsp;特征图大小为62x37，所以会一共生成 **62 x 37 x 9=20646** 个Anchor box。

&emsp;&emsp;源码中，通过width:(0~62)x16, height(0~37)x16建立shift偏移量数组，再和base_anchor基准坐标数组累加，得到特征图上所有像素对应的Anchors的坐标值，是一个[20646, 4]的数组

&emsp;&emsp;注：通过中心点和size就可以得到滑窗位置和原图位置的映射关系，由此原图位置并根据与Ground Truth重复率贴上正负标签，让RPN学习该Anchors是否有物体即可；另外，论文中也提到了，相比于只采用单一尺度和长宽比，单尺度多长宽比和多尺度单长宽比都能提升mAP，表明多size的anchors可以提高mAP，作者在这里选取了最高mAP的3种尺度和3种长宽比）

&emsp;&emsp;通过增加一个3x3滑动窗口操作以及两个卷积层1x1卷积层完成区域建议功能；对得分区域进行非极大值抑制后输出得分Top-N（文中为300）区域，告诉检测网络应该注意哪些区域，本质上实现了Selective Search、EdgeBoxes等方法的功能。

&emsp;&emsp;Faster R-CNN使用RPN生成候选框后，剩下的网络结构和Fast R-CNN中的结构一模一样。在训练过程中，需要训练两个网络，一个是RPN网络，一个是在得到框之后使用的分类网络。通常的做法是交替训练，即在一个batch内，先训练RPN网络一次，再训练分类网络一次。

- **bounding box regression原理**：详见<https://zhuanlan.zhihu.com/p/31426458>



#### 2.2.2) rpn-data （**训练RPN**）

```python
layer 
{  
  name: 'rpn-data'  
  type: 'Python'  
  bottom: 'rpn_cls_score'   #仅提供特征图的height和width的参数大小
  bottom: 'gt_boxes'        #ground truth box
  bottom: 'im_info'         #包含图片大小和缩放比例，可供过滤anchor box
  bottom: 'data'  
  top: 'rpn_labels'  
  top: 'rpn_bbox_targets'  
  top: 'rpn_bbox_inside_weights'  
  top: 'rpn_bbox_outside_weights'  
  python_param 
  {  
    module: 'rpn.anchor_target_layer'  
    layer: 'AnchorTargetLayer'  
    param_str: "'feat_stride': 16 \n'scales': !!python/tuple [8, 16, 32]"  
  }  
}
```

&emsp;&emsp;接下来RPN做的事情就是利用（`AnchorTargetCreator`）将20000多个候选的anchor选出256个anchor进行分类和回归位置。选择过程如下：

- 对于每一个ground truth bounding box (`gt_bbox`)，选择和它**重叠度（IoU）最高的一个anchor作为正样本**
- 对于剩下的anchor，从中选择和任意一个 `gt_bbox` **重叠度超过0.7的anchor**，作为正样本，正样本的数目不超过128个。
- 随机选择和 `gt_bbox ` **重叠度小于0.3的anchor作为负样本**。负样本和正样本的总数为256；如果正样本不足128，则增加一些负样本。

对于每个anchor, gt_label 要么为1（前景），要么为0（背景），而gt_loc则是由4个位置参数(tx,ty,tw,th)组成，即**计算anchor box与ground truth之间的偏移量，这样比直接回归坐标更好**。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/8_0.png)

计算分类损失用的是交叉熵损失，对应于‘rpn_loss_cls’；而计算回归损失用的是Smooth_l1_loss，对应于‘rpn_loss_bbox’，在计算回归损失的时候只计算正样本（前景）的损失，不计算负样本的位置损失。‘rpn_cls_prob’计算概率值(可用于下一层的nms非最大值抑制操作)



#### 2.2.3) proposal **（RPN生成RoIs）**

```python
layer 
{  
   name: 'proposal'  
   type: 'Python'  
   bottom: 'rpn_cls_prob_reshape' #[1,18,40,60]==> [batch_size, channel，height，                  width]Caffe的数据格式，anchor box分类的概率
   bottom: 'rpn_bbox_pred'  # 记录训练好的四个回归值△x, △y, △w, △h
   bottom: 'im_info'  
   top: 'rpn_rois'  
   python_param 
   {  
     module: 'rpn.proposal_layer'  
     layer: 'ProposalLayer'  
     param_str: "'feat_stride': 16 \n'scales': !!python/tuple [4, 8, 16, 32]"
   }  
 }
```

RPN在自身训练的同时，还会提供RoIs（region of interests）给Fast RCNN（RoIHead）作为训练样本。RPN生成RoIs的过程(`ProposalCreator`)如下：

- 对于每张图片，利用它的feature map， 计算 (H/16)× (W/16)×9（大概20000）个anchor属于前景的概率，以及对应的位置参数。
- 选取概率较大的12000个anchor；
- 利用回归的位置参数（在输入中我们看到’rpn_bbox_pred’，记录着训练好的四个回归值△x, △y, △w, △h，累加上训练好的△x, △y, △w, △h，从而得到了相较于之前更加准确的预测框region proposal），修正这12000个anchor的位置，得到RoIs；
- 进一步对RoIs进行越界剔除（训练阶段），和非极大值抑制（Non-maximum suppression, NMS，一般取0.7），选出概率最大的2000个RoIs

注意：在inference的时候，为了提高处理速度，12000和2000分别变为6000和300。

注意：这部分的操作不需要进行反向传播，因此可以利用numpy/tensor实现。

RPN的输出：RoIs（形如2000×4或者300×4的tensor）



用下图一个案例来对NMS算法进行简单介绍

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/10.png)

如上图所示，一共有6个识别为人的框，每一个框有一个置信率。 

现在需要消除多余的:

·     按置信率排序: 0.95, 0.9, 0.9, 0.8, 0.7, 0.7

·     取最大0.95的框为一个物体框

·     剩余5个框中，去掉与0.95框重叠率IoU大于0.6(可以另行设置)，则保留0.9, 0.8, 0.7三个框

·     重复上面的步骤，直到没有框了，0.9为一个框

·     选出来的为: 0.95, 0.9

所以，整个过程，可以用下图形象的表示出来：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/11.png)

其中，红色的A框是生成的anchor box,而蓝色的G’框就是经过RPN网络训练后得到的较精确的预测框，绿色的G是ground truth box。

&emsp;&emsp;

#### 2.2.4) roi_data 

```python
 layer 
 {  
  name: 'roi-data'  
  type: 'Python'  
  bottom: 'rpn_rois'  
  bottom: 'gt_boxes'  
  top: 'rois'  
  top: 'labels'  
  top: 'bbox_targets'  
  top: 'bbox_inside_weights'  
  top: 'bbox_outside_weights'  
  python_param 
  {  
     module: 'rpn.proposal_target_layer'  
     layer: 'ProposalTargetLayer'  
     param_str: "'num_classes': 81"  
  }  
}  
```

**RoIHead网络结构**

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/14_1.jpg)

为了避免定义上的误解，我们将经过‘proposal’后的预测框称为region proposal（其实，RPN层的任务其实已经完成，roi_data属于为下一层准备数据）

主要作用：

①       RPN层只是来确定region proposal是否是物体(是/否)，这里根据region proposal和ground truth box的最大重叠指定具体的标签(就不再是二分类问题了，参数中指定的是81类)

②       计算region proposal与ground truth boxes的偏移量，计算方法和之前的偏移量计算公式相同

经过这一步后的数据输入到RoI Pooling层进行进一步的分类和定位.



**ROI Pooling:**

```python
layer 
{  
  name: "roi_pool5"  
  type: "ROIPooling"  
  bottom: "conv5_3"   #输入特征图大小
  bottom: "rois"      #输入region proposal
  top: "pool5"     #输出固定大小的feature map
  roi_pooling_param 
  {  
    pooled_w: 7  
    pooled_h: 7  
     spatial_scale: 0.0625 # 1/16     
  }  
}
```

从上述的Caffe代码中可以看到，输入的是RPN层产生的region proposal（假定有300个region proposal，这个值应该是一个超参数？给出得分Top300的region proposal？）和VGG16最后一层产生的特征图（62 x 37 x 512-d），遍历每个region proposal，将其坐标值缩小16倍，这样就可以将在原图(1000 x 600)基础上产生的region proposal映射到 62 x 37 的特征图上，从而将在feature map上确定一个区域(定义为RB * )。

在feature map上确定的区域RB * ，根据参数pooled_w:7,pooled_h:7,将这个RB * 区域划分为7 x 7，即49个相同大小的小区域，对于每个小区域，使用max pooling方式从中选取最大的像素点作为输出，这样，就形成了一个7 x 7的feature map；

​       细节可查看：<https://www.cnblogs.com/wangyong/p/8523814.html>

以此，参照上述方法，300个region proposal遍历完后，会产生很多个7*7大小的feature map，故而输出的数组是：[300,512,7,7],作为下一层的全连接的输入。

详细训练过程可以参考：<https://blog.csdn.net/u011956147/article/details/53053381#commentBox>

<https://blog.csdn.net/github_37973614/article/details/81258593#commentBox>



**全连接层:**

经过roi pooling层之后，batch_size=300, proposal feature map的大小是7*7,512-d,对特征图进行全连接，参照下图，最后同样利用Softmax Loss和L1 Loss完成分类和定位

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/FasterRCNN/15.png)

通过full connect层与softmax计算每个region proposal具体属于哪个类别（如人，马，车等），输出cls_prob概率向量；同时再次利用bounding box regression获得每个region proposal的位置偏移量bbox_pred，用于回归获得更加精确的目标检测框

即从PoI Pooling获取到7x7大小的proposal feature maps后，通过全连接主要做了：

4.1)通过全连接和softmax对region proposals进行具体类别的分类

4.2)再次对region proposals进行bounding box regression，获取更高精度的rectangle box



##  三、训练过程：

- Step 1：用model初始化RPN网络，然后训练RPN，在训练后，model以及RPN的unique会被更新。

- Step 2：用model初始化Fast-rcnn网络，注意这个model和第一步一样。然后使用训练过的RPN来计算proposal，再将proposal给予Fast-rcnn网络。接着训练Fast-rcnn。训练完以后，model以及Fast-rcnn的unique都会被更新。说明：第一和第二步，用同样的model初始化RPN网络和Fast-rcnn网络，然后各自独立地进行训练，所以训练后，各自对model的更新一定是不一样的（论文中的different ways），因此就意味着model是不共享的（论文中的dont share convolution layers）。

- Step 3：使用第二步训练完成的model来初始化RPN网络，第二次训练RPN网络。但是这次要把model锁定，训练过程中，model始终保持不变，而RPN的unique会被改变。说明：因为这一次的训练过程中，model始终保持和上一步Fast-rcnn中model一致，所以就称之为共享。

- Step 4：仍然保持第三步的model不变，初始化Fast-rcnn，第二次训练Fast-rcnn网络。其实就是对其unique进行finetune，训练完毕，得到一个文中所说的unified network。



## 四、其他问题

- 如何处理多尺度多长宽比问题？即如何使24×24和1080×720的车辆同时在一个训练好的网络中都能正确识别？

  论文中展示了两种解决多尺度多长宽比问题：一种是使用图像金字塔，对伸缩到不同size的输入图像进行特征提取，虽然有效但是费时；另一种是使用滤波器金字塔或者滑动窗口金字塔，对输入图像采用不同size的滤波器分别进行卷积操作，这两种方式都需要枚举图像或者滤波器size；作者提出了一种叫Anchors金字塔的方法来解决多尺度多长宽比的问题，在RPN网络中对特征图滑窗时，对滑窗位置中心进行多尺度多长宽比的采样，并对多尺度多长宽比的anchor boxes区域进行回归和分类，利用Anchors金字塔就仅仅依赖于单一尺度的图像和特征图和单一大小的卷积核，就可以解决多尺度多长宽比问题，这种对推荐区域采样的模型不管是速度还是准确率都能取得很好的性能。


- 同传统滑窗方法提取区域建议方法相比，RPN网络有什么优势？

  传统方法是训练一个能检测物体的网络，然后对整张图片进行滑窗判断，由于无法判断区域建议的尺度和长宽比，所以需要多次缩放，这样找出一张图片有物体的区域就会很慢；虽然RPN网络也是用滑动窗口策略，但是滑动窗口实在卷积层特征图上进行的，维度较原始图像降低了很多倍【中间进行了多次max pooling 操作】,RPN采取了9种不同尺度不同长宽比的anchors，同时最后进行了bounding-box回归，即使是这9种anchors外的区域也能得到一个跟目标比较接近的区域建议。


- 文中anchors的数目？

  文中提到对于1000×600的一张图像，大约有20000 (~62×37×9) 个anchors，忽略超出边界的anchors剩下6000个anchors，利用非极大值抑制去掉重叠区域，剩2000个区域建议用于训练；测试时在2000个区域建议中选择Top-N（文中为300）个区域建议用于Fast R-CNN检测。


- hard negative mining（ 难分样本挖掘）是如何实现的？

  链接：https://www.zhihu.com/question/46292829/answer/628723483

  网友1：在训练分类器的时候，对不同的类别，都需要一定的正样本，这个正样本的数据集来源就是，生成的proposals与某个标注数据的ground true重叠区域（IoU）大于某个阈值，这个proposal才会被作为这个类别的正样本。在Fast R-CNN中，proposals是由Selective Search算法给出的，这些proposals里包含有“背景”，和“其他物体”，值得注意的是，这个“其他物体”不一定是我们训练的类别。所以这就会产生一个问题，对于一副图像，我们要检测“人”，但实际图像中只有一个或是两个人。虽然我们生成了不少（~2k）的proposals，但这些proposals里跟人的标注数据的ground true的IoU大于某个阈值的（比如0.5），其实并不多。因为proposals不是专门为某个类别（这里的例子是“人”）而生成的。负样本过多会造成，正样本大概率被预测为负样本。因此**作者使用随机抽样的方式，抽取25%正样本，75%的负样本**。（注：为何是1:3, 而不是1:1呢? 可能是正样本太少了, **如果 1:1 的话, 一张图片处理的 RoI 就太少了, 不能充分的利用 RoI Pooling 前一层的共享计算, 训练的效率就太低了, 但是负例比例也不能太高了, 否则算法会出现上面所说的 false negative 太多的现象, 选择 1:3 这个比例是算法在性能和效率上的一个折中考虑, 同时 OHEM(online hard example mining)一文中也提到负例比例设的太大, Fast RCNN 的 mAP将有很大的下降**  https://www.cnblogs.com/nowgood/p/Hardexamplemining.html ）

  **但为什么要设置proposals【0.1<=IoU<0.5】为负样本，而proposals【IoU<0.1】作为难样本挖掘(hard negative mining)呢？不是要拿proposals【0.1<=IoU<0.5】这些容易分错的来做难样本挖掘吗**？其实按道理应该从源码上去看作者是怎么实现的，不过由于我还没看，我先给出我一个自己的想法。

  proposals【0.1<=IoU<0.5】实际上已经是作为hard negative去训练了，因为负样本的随机抽样就是从这里面抽取的。但这样的样本可能不多。

  而proposals【IoU<0.1】，这些样本数量比较多，而里面可能也会有让分类器误判的样本。当我们第一轮用proposals【0.1<=IoU<0.5】和【IoU>=0.5】抽样的样本，训练出来的模型，去预测proposals【IoU<0.1】的样本，如果判断错误就加入hard negative的集合里，这样就实现了对proposals【IoU<0.1】的hard negatvie mining。



  网友2：**我们可以先验的认为, 如果 RoI 里没有物体，全是背景，这时候分类器很容易正确分类成背景，这个就叫 easy negative, 如果RoI里有二分之一个物体，标签仍是负样本，这时候分类器就容易把他看成正样本，这时候就是 hard negative**。

  确实, 不是一个框中背景和物体越混杂, 越难区分吗? 框中都基本没有物体特征, 不是很容易区分吗?

  那么我认为 Fast RCNN 也正是这样做的, 为了解决正负样本不均衡的问题（负例太多了）, 我们应该剔除掉一些容易分类负例, 那么与 ground truth 的 IOU 在 [0, 0.1)之间的由于包含物体的特征很少, 应该是很容易分类的, 也就是说是 easy negitive, 为了让算法能够更加有效, 也就是说让算法更加专注于 hard negitive examples, 我们认为 hard negitive examples 包含在[0.1, 0.5) 的可能性很大, 所以训练时, 我们就在[0.1, 0.5)区间做 random sampling, 选择负例.

  **我们先验的认为 IoU 在[0, 0.1)之内的是 easy example, 但是, [0, 0.1) 中包含 hard negitive examples 的可能性并非没有, 所以我们需要对其做 hard negitive mining, 找到其中的 hard negitive examples 用于训练网络**. **按照常理来说 IOU 在[0, 0.1)之内 会被判定为真例的概率很小, 如果这种现象发生了, 可能对于我们训练网络有很大的帮助, 所以 Fast RCNN 会对与 ground truth 的 IoU 在 [0, 0.1)之内的是 example 做 hard negitive examples**.



  网友3：**hard negative就是每次把那些顽固的棘手的错误,再送回去继续练,练到你的成绩不再提升为止。**





主要参考：

<https://blog.csdn.net/wopawn/article/details/52223282>

<https://zhuanlan.zhihu.com/p/32404424>
