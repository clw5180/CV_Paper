# **Anchor**

参考：https://zhuanlan.zhihu.com/p/63273342

&emsp;&emsp;&emsp;https://zhuanlan.zhihu.com/p/68291859



## 一、基本概念

最近一段时间各种所谓anchor-free的detection算法得到了很多的关注。这里先来谈谈各种对于anchor based detection方法中anchor设置的改进。先来介绍一些anchor的概念，以及它在detection系统中发挥的作用。

**第一个观点是，绝大多数top-down detector都是在做某种意义上的refinement和cascade。**区别无外乎在于，refine的次数以及每次refine的方法。在传统方法中从无论sliding window出发，不断去筛选正样本；还是后续使用Selective search或者Edgebox直接生成proposal都是基于这样的思路。后续基于Deep learning的方法也没有走出这个套路，在one stage算法中，主流的方法便是在refine预先指定的anchor；在two stage算法中，试图使用RPN来替代sliding window或者其他生成proposal的方法，再通过提取的region feature来refine这些proposal。

anchor这个概念最早出现在Faster RCNN的paper中，如果能理解前面提到的内容，其实anchor就是在这个cascade过程中的起点。由于在Faster RCNN那个年代还没有FPN这种显式处理scale variation的办法，**anchor的一大作用便是显式枚举出不同的scale和aspect ratio**。原因也很简单，只使用一个scale的feature map和同一组weight，要去预测出所有scale和aspect ratio的目标，本身就是很困难的一件事。通过anchor的引入，将scale和aspect ratio进行划分，针对某个特定的区间和组合，使用一组特定学习到的weight去处理，从而缓解这个问题。需要注意的是，**anchor本身并不会参与到网络的运算中去，影响的只会是classification和bbox regression分支的target（训练阶段）和怎样decode box（测试阶段）**。换句话说，网络其实预测的是相对于anchor的offset，只有在最终从offset转换到bbox时，才会使用。这样的想法也很自然被各种One stage方法所吸收，形成了anchor已经是detection标配的stereotype。

一般而言，**对于anchor shape的设定，除了手工拍拍脑袋随意设置几个scale和aspect ratio之外，对于ground-truth bbox进行一次聚类**也是一个常用的方法。

另外，对于anchor的base size，scale和ratio的含义，大佬martinzlocha给出了说明：“The ratios are slightly harder to compute but the size of the final anchor is sizes x scales. So essentially if you have size 32x32 and scale of 0.5 then the anchor will be 16x16.If you have a ratio of 1:2 and the (1:1) anchor is **16x16** then the **1:2** anchor is xxy=16x16 where x/y=2. So just solve the two simultaneous equations and you will get **11.31x22.62**”



## 二、典例

- 关于RetinaNet中，求anchor的偏移（shift）和中心（center），有网友对RetinaNet的复现者fizyr进行提问 https://github.com/fizyr/keras-retinanet/issues/1073

  相关代码：

  ```python
  def shift(shape, stride, anchors):
      # shape  : Shape to shift the anchors over.
      # stride : Stride to shift the anchors with over the shape.
      shift_x = (np.arange(0, shape[1]) + 0.5) * stride
      shift_y = (np.arange(0, shape[0]) + 0.5) * stride
      
      # top_left_offset =  (H - (h - 1) * stride - 1) // 2.0, (W - (w - 1) * stride - 1) // 2.0
  ```

  这里作者举了个例子，比如输入图片尺寸是256x256，此时P3对应的feature map大小是32x32，feature map上的每个pixel的stride为8。由于P3层的anchor base size是32x32（映射到输入原图），因此感受野的大小为32 + 31 x 8 = 280，比原图多了280 - 256 = 24，因此anchor应该偏移24 / 2 = 12 pixels；公式：top_left_offset =  (H - (size + (h - 1) * stride)) / 2.0, (W - (size + (w - 1) * stride - 1)) / 2.0，其中比如(H - (size + (h - 1) * stride)) / 2.0 = (256 - (32 + (32 - 1) * 8)) / 2 = -12。这里size就是anchor的base size，P3层是32。

  也就是说，如果P3层feature map上的pixel从(0, 0)移动到了(1, 0) （自注：这里讨论的是左上角的坐标），那么base anchor就会相应地从原图的(-12, -12)移动到(-4, -12)，相当于水平方向移动了一个stride，也就是8；另外，因为feature map上左上角(0, 0)这个pixel对应于原图的(-12, -12)这个位置，因此可以推出中心位于原图的(4, 4)位置，如下图所示：

  ![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/Anchor/1.png)

  

  网友提出的质疑：

  “So in your example (input 800x800):
  P3 has shape 100x100 and stride 8x8. Hence the offset should be (800 -
  8*99)/2 = 4
  P4 has shape 50x50 and stride 16x16. Hence the offset should be (800 -
  16*49)/2 = 8
  P5 has shape 25x25 and stride 32x32.  Hence the offset should be (800 -
  32*24)/2 = 16
  P6 has shape 13x13 and stride 64x64.  Hence the offset should be (800 -
  64*12)/2 = 16
  P7 has shape 7x7 and stride 128x128.  Hence the offset should be (800 -
  128*6)/2 = 16
  So as you can see, the reason is that in P6 and P7 we did an imperfect
  "max-pooling":
  Each one of the 13x13 pixels should cover 2x2 pixels from the previous
  25x25 layer, so this means that we implicitly added a padding of 1
  somewhere (because 13x13 covers 26x26 area).
  The question is, where?  This computation assumes that we added the padding
  symmetrically (i.e. added 0.5 from each side).
  But I don't think maxpooling can do that.
  So I'm guessing the maxpooling appends the 25x25 layer with another column
  and row of zeros somewhere. Probably on the bottom, probably on the right.
  This breaks the symmetry.
  If feature (0,0) in P5 was centered at (16, 16), and the feature (1,1) was
  centered at (48, 48), (because in P5 stride=32), then after we maxpool
  features (0,0), (0,1), (1,0), (1,1), the center should move to (32,32).
  So you're right - this computation is wrong, assuming max-pooling adds an
  asymmetric padding.”



- 检测小目标，比如使用P2到P6层，大佬martinzlocha给出的建议：

  If you want to use 16-256 and half the stride then you need P2-P6. To use P2-P6 just modify the line: [https://github.com/fizyr/keras-retinanet/blob/50d07a0cd7c6e019413bfa3f4c7d7159401e8747/keras_retinanet/models/vgg.py#L97](https://slack-redir.net/link?url=https%3A%2F%2Fgithub.com%2Ffizyr%2Fkeras-retinanet%2Fblob%2F50d07a0cd7c6e019413bfa3f4c7d7159401e8747%2Fkeras_retinanet%2Fmodels%2Fvgg.py%23L97) so that you use the higher pooling layers. So starting from block2_pool.

  You'll also want to change a few small things like this: [https://github.com/fizyr/keras-retinanet/blob/50d07a0cd7c6e019413bfa3f4c7d7159401e8747/keras_retinanet/utils/anchors.py#L220](https://slack-redir.net/link?url=https%3A%2F%2Fgithub.com%2Ffizyr%2Fkeras-retinanet%2Fblob%2F50d07a0cd7c6e019413bfa3f4c7d7159401e8747%2Fkeras_retinanet%2Futils%2Fanchors.py%23L220)

  And you cannot just arbitrarily change the stride. Stride is the ratio between the feature map size to which you attach the subnetworks and the original image size.

  However you might get better results just by dividing the scales values by half instead of shifting everything because you are making the network shallower effectively.

  即，不能简单地把anchor base size从[32, 64, 128, 256, 512]改成[16, 32, 64, 128, 256]，还需要对应到具体的backbone层，把regression和classification子网络从P3~P7改到P2~P6；而且strides最好不要改，应为反映了feature map size到原图size的一个关系；或许改anchor scales改成0.5会好一点。

