# **Arbitrary-Oriented Scene Text Detection via Rotation Proposals**

论文地址：<https://arxiv.org/abs/1703.01086>

代码复现：<https://github.com/mjq11302010044/RRPN_pytorch> （PyTorch版本）



## 一、介绍

RRPN沿用了Faster-rcnn中的RPN的思想(即使用其来生成候选区域)，不同的地方是：
**1、提出了基于旋转候选网络区域(RRPN)**
**2、角度回归**


## 二、主要内容

- 前端使用VGG16作为特征提取主干网络
- 中间采用RRPN主要是用于生成带倾斜角的候选区域，该层输出包括候选框的类别和旋转矩形框的回归
- 将RRPN生成的候选框映射到特征图上，得到最终的文本行检测结果

具体的网络结构图如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RRPN/1.png)



####  具体实现细节

##### 1、旋转矩形框的表示（暂略，参考<https://zhuanlan.zhihu.com/p/39717302>）

数据的表示形式为 (x_ctr, y_ctr, h, w, θ)，这个角度的范围是(-45°, 135°]，使用这样的五个变量进行表示具有3个好处：
- **1）能够比较容易计算出两个旋转的角度差异；**
- **2）相较于传统上对于边界框使用8个点的表达，使用这样的方式更能很好回归带有旋转的目标检测框情形；**
- **3）使用这样的表达能够高效计算经过旋转之后训练图片的ground truth；**


##### 2、旋转anchor

相比传统的anchors，本文设计的旋转anchor（简称R-anchor）进行了如下的改进：1) 通过加入方向参数来控制proposal的方向，这里结合方向收敛速度和计算效率选用了6个方向，分别是 （-30°，0，30°，60°，90°，120°）； 2)修改了文本行的aspect ratio，调整文本行的比例为1:2，1:5，1:8，大小还是8，16和32，具体的anchor策略示意图如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RRPN/1.png)

结合上述的aspect ratio和scale，特征图上每个点将生成**54个R-anchor**（6个方向，3个尺度，3个宽高比）．对应的回归层有 54x5=270 个输出值，分类层有 54x2=108 个输出值。对于大小为 HxW 的特征图，通过RRPN可以产生 **H x W x 54** 个R-anchor。



##### 3、确定正负样本的依据

如果同时满足以下两个条件则会被标记为正样本：
1）与ground truth的IoU最大的，或者IoU大于0.7    
2）与ground truth的夹角小于15° 

如果满足以下条件之一则会被标记为负样本：
1) 与ground truth的IoU小于0.3
2) 其与ground truth的IoU大于0.7，但其与ground truth的夹角大于15°

对那些既不满足正样本也不满足负样本的候选区域，不参与训练。



##### 4、损失函数（暂略）
损失函数采用的是多任务损失函数。


##### 5、候选框精修（暂略）
**（1）倾斜框IOU计算**

**（2）倾斜NMS**


##### 6、RRoI Pooling层（暂略）






## 三、实验结果

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/RRPN/3.png)

&emsp;&emsp;在实际训练中，其召回率比较低。



## 四、结论

* 
* 



参考：<https://zhuanlan.zhihu.com/p/39717302>
