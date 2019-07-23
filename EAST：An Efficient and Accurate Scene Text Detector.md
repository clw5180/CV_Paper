论文：EAST：An Efficient and Accurate Scene Text Detector
============

论文地址：https://arxiv.org/pdf/1704.03155.pdf

代码复现：https://github.com/argman/EAST



## 1. 介绍

&emsp;&emsp;该模型直接预测全图像中任意方向和四边形形状的单词或文本行，**消除了很多中间过程**（例如候选区域聚合和单词分割等)，直接预测文本行，从而**减少检测时间**，同时保证不错的准确度，很好地实现了端到端的文本检测。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/1-0.png)



## 2. 主要内容

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/1.png)

#### 第一部分：Feature extractor stem(PVANet)

- 先用一个通用的网络(论文中采用的是**PVANet**，实际在使用的时候可以采用VGG16，Resnet等)作为base net ，基于上述主干特征提取网络，提取不同level的不同尺度的feature map（它们的尺寸分别是input-image的1/32， 1/16， 1/8， 1/4）并用于后期的特征组合（concatenate），目的是解决文本行尺度变换剧烈的问题；检测大块区域的文本需要神经网络后级的高阶特征，而检测小块区域的文本需要前级的低阶信息。
- 代码描述： 

```python
# 首先是一个resnet_v1_50网络
with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
    logits, end_points = resnet_v1.resnet_v1_50(images, 
                                                is_training=is_training, 															scope='resnet_v1_50')
	with tf.variable_scope('feature_fusion', values=[end_points.values]):
    batch_norm_params = 
    {
   	 'decay': 0.997,
   	 'epsilon': 1e-5,
   	 'scale': True,
    	'is_training': is_training
    }
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu, # 激活函数是relu
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(weight_decay)):# L2正则
        f = [end_points['pool5'], end_points['pool4'],
             end_points['pool3'], end_points['pool2']]
```



#### 第二部分：Feature-merging branch

- 在这一部分用来组合特征，并通过**上池化和concat**恢复到原图的尺寸，这里借鉴的是**U-net**的思想。 
所谓上池化一般是指最大池化的逆过程，实际上是不能实现的但是，可以通过只把池化过程中最大激活值所在的位置激活，其他位置设为0，完成上池化的近似过程。 g和h的计算过程如下图所示。 

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/2.png)

- 代码描述：

```python
for i in range(4):
       print('Shape of f_{} {}'.format(i, f[i].shape))
   g = [None, None, None, None]
   h = [None, None, None, None]
   num_outputs = [None, 128, 64, 32]
   for i in range(4):
       if i == 0:
           h[i] = f[i]  # 计算h
       else:
           c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
           h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
       if i <= 2:
           g[i] = unpool(h[i]) # 计算g
       else:
           g[i] = slim.conv2d(h[i], num_outputs[i], 3)
       print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
```



#### 第三部分：Output Layer

- 上一部分的输出通过一个（1x1，1）的卷积核获得score_map。**score_map与原图尺寸一致**，**每一个值代表此处是否有文字的可能性**。 
- 上一部分的输出通过一个（1x1，4）的卷积核获得**RBOX** 的geometry_map。有四个通道，**分别代表每个像素点到文本矩形框上，右，底，左边界的距离**。另外再通过一个（1x1, 1）的卷积核获得该框的旋转角，这是为了能够识别出有旋转的文字。 
- 上一部分的输出通过一个（1x1，8）的卷积核获得QUAD的geometry_map，八个通道分别代表每个像素点到任意四边形的四个顶点的距离。 

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/2-1.png)

- 代码描述：

```python
# 计算score_map
F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
# 4 channel of axis aligned bbox and 1 channel rotation angle
# 计算RBOX的geometry_map
geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
# angle is between [-45, 45] #计算angle
angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2
F_geometry = tf.concat([geo_map, angle_map], axis=-1)
```



#### 训练标签生成
QUAD的分数图生成与几何形状图生成（暂略，详见<https://zhuanlan.zhihu.com/p/37504120>）



#### 损失函数
损失函数分为2个部分：（1）分类误差（score map loss）  （2）几何误差（geometry loss），文中权衡重要性，λg=1。 

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/3.png)



#### 分类误差函数（score map loss）

论文中采用的是类平衡交叉熵（class-balanced cross-entropy），用于解决正负样本不均衡的问题（how？）。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/4.png)

β=反例样本数量/总样本数量 （balance factor） 

- 代码描述：

  ```python
  # 计算score map的loss
  def dice_coefficient(y_true_cls, y_pred_cls,
                       training_mask):
      '''
      dice loss
      :param y_true_cls:
      :param y_pred_cls:
      :param training_mask:
      :return:
      '''
      eps = 1e-5
      intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
      union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
      loss = 1. - (2 * intersection / union)
      tf.summary.scalar('classification_dice_loss', loss)
      return loss
  
  ```

但是在具体实战中，一般采用dice loss，它的收敛速度会比类平衡交叉熵快；



#### 几何误差函数（geometry loss）

几何形状损失
文本在自然场景中的尺寸变化极大。直接使用L1或者L2损失去回归文本区域将导致损失偏差朝更大更长．因此论文中采用IoU损失在RBOX回归的AABB部分，尺度归一化的smoothed-L1损失在QUAD回归，来保证几何形状的回归损失是尺度不变的．

对于RBOX，采用IoU loss ：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/5.png)

角度误差则为： 

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/6.png)

对于QUAD采用smoothed L1 loss 
CQ={x1,y1,x2,y2,x3,y3,x4,y4} 

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/7.png)

NQ*指的是四边形最短边的长度 。
- 代码描述：

```python
def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union  #计算R_true与R_pred的交集
    area_union = area_gt + area_pred - area_intersect  #计算R_true与R_pred的并集
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0)) # IoU loss,加1为了防止交集为0，log0没意义
    L_theta = 1 - tf.cos(theta_pred - theta_gt) # 夹角的loss
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta # geometry_map loss

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
```



#### 采用局部感知NMS（locality-aware NMS）进行几何过滤

由于本文中面临的是成千上万个几何体，如果用普通的NMS，其计算复杂度是O(n^2)，n是几何体的个数，这是不可接受的。本文在假设来自附近像素的几何图形倾向于高度相关的情况下，逐行合并几何图形，并且在合并同一行中的几何图形时将迭代合并当前遇到的几何图形。

测试结果：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/7-1.png)

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/7-2.png)

缺点： 
由于感受野不够大，所以对于较长的文字比较难识别。



#### 训练其它参数
使用Adam优化器对网络展开端到端的训练；对输入图像进行裁剪，为了加速训练过程，crop到512*512大小，mini-batch size=24，初始lr=1e-3，每经过27300个mini-batch学习率降低为原来的1/10，并在lr=1e-5时停止继续减小。具体实验过程中可以使用linear learning rate decay。



## 3. 训练结果

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/EAST/8.png)



## 4. 总结

- 在特征合并层，**利用不同尺度的feature map**，并通过相应的规则进行**自顶向下的合并方式，可以检测不同尺度的文本行**；
- 提供了文本的方向信息，**可以检测各个方向的文本**
- 本文的方法**在检测长文本的时候效果表现比较差，这主要是由网络的感受野不够大导致的**
- 在检测曲线文本时，效果不太理想

主要参考：<https://blog.csdn.net/zhangwei15hh/article/details/79899300>
