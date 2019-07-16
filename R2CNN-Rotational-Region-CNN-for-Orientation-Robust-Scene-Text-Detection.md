

# **R2CNN-Rotational-Region-CNN-for-Orientation-Robust-Scene-Text-Detection**

论文发布日期：2017.6.29  [CVPR]
论文链接：https://arxiv.org/abs/1706.09579

## 1. Introduction  
&emsp;&emsp;文本检测的**特点和挑战** ：尺度变化性、特殊的宽高比例、方向、不同字体风格、光照、角度等，其中尤其是**方向检测**十分重要。这篇文章提出了**R2CNN（Rotational Region CNN**）算法解决**旋转文本**的检测。R2CNN算法的整体结构如下图 ：


* 本文方案

![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/1.png)

&emsp;&emsp;主要是在Faster RCNN算法的基础上做了一些修改：

​	1、RoI Pooling时的尺寸除了7×7外，还有两种长宽不一致的尺寸：**3×11和11×3**，这种设置的用意也非常明显，就是为了**解决水平和竖直长文本的检测**。然后对于提取到的RoI特征做concat操作进行融合，作为后续预测支路的输入。

​	2、预测输出有3个支路，第一个支路是有无文本的二分类，这个和目标检测算法中的目标分类类似。第二个支路是水平框（axis-aligned box）的预测，这个和目标检测算法中的框预测一样。第三个支路时倾斜框（inclined box）的预测，这部分是这篇文章的亮点，而且该支路后面跟一个NMS进行处理得到最后结果。至于RPN网络部分输出的RoI则和常规目标检测中RPN网络输出的RoI一样，都是水平方向。另外这篇文章其实还**增加了一些小尺寸的anchor提升对小文本的检测效果**。所以R2CNN算法最后既有常规的水平预测框输出，也有倾斜框输出，这两种框都是基于RPN网络输出的RoI得到的，虽然倾斜框支路也能预测水平框，但是作者认为第二个支路的存在对最后结果帮助较大。

&emsp;&emsp;综上，简单来说就是**首先RPN生成正常的水平proposal（~~论文用的axis-aligned，应该是表述不清，实际是水平竖直检测框~~），二分类筛掉，然后保留的每个proposal回归斜框和水平框坐标，最后对斜框进行斜向NMS得到最终检测结果**。效果如下图：        

​![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/2.png)

&emsp;&emsp;**因此R2CNN算法整体上的处理流程可以用上图所示**。a是原图；b是RPN生成的RoI，这些ROI区域都是常规的水平框；c是回归得到的水平框（第二个支路）和倾斜框（第三个支路）；d是经过倾斜框的NMS算法后得到的最终输出结果。注意R2CNN和RRPN的区别，**RRPN是直接在RPN阶段就生成斜框了**。




## 2. Proposed Approach

### 2.1 Problem definition
* gt定义 
&emsp;&emsp;文本检测的ICDAR竞赛数据集是不规则的四边形，用顺时针的四个点坐标进行表征，不是严格的矩形框。这种不规则四边形一般可以用斜矩形来近似拟合，所以后面的bbox都采用斜矩形进行bbox预测。

* 旋转表征 
&emsp;&emsp;关于旋转方向的表征，很多方法是直接选择的角度回归来直接表征， 比如可以用Figure3(a)即下图所示的4个点坐标来表示一个任意形状的四边形，这种表示方式其实就覆盖了水平框和倾斜框，而且框的形状不仅限于矩形，而是延伸至四边形。但这篇文章不采用任意形状的四边形预测方式，而认为倾斜矩形框足够覆盖待检测的文本，**因此这篇文章所提到的倾斜框是指倾斜的矩形框，为了描述简单，后续都用倾斜框代替倾斜的矩形框**。倾斜框的定义也有好几种，比如可以用水平框+旋转角度来表示一个倾斜框，这种表示方式在旋转目标检测算法中也比较常见，比如EAST算法。但是这篇文章认为训练旋转角度时在有些角度情况下模型不是很稳定，主要是因为一些特殊的相似角度在表示上有可能差别较大，比如90度和-90度，**所以考虑采用Figure3(b)(c)这种用两个相邻点坐标和高来定义一个倾斜框**。
![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/3.png)



### 2.2 Rotational Region CNN (R2CNN)

* RPN for proposing axis-aligned boxes  
  &emsp;&emsp;RPN和常规的相似，加了更小的anchor尺寸以便获得更好的效果。其他配置均同Faster R-CNN。
* RoIPoolings of different pooled sizes  
  &emsp;&emsp;为了适应文本不同方向的特殊长宽比，除了标准7x7外，还采用了不同比例的RoIpooling为11x3和3x11的信息进行获取融合
* Regression for text/non-text scores, axis-aligned boxes, and inclined minimum area boxes  
  &emsp;&emsp;在回归上，<u>不仅回归斜框坐标，还回归正框</u>，实验表明这样的效果能提高。
* Inclined non-maximum suppression    
  &emsp;&emsp;实验表明，斜的NMS效果会比正的好。很好理解，对于<u>斜向密集目标的检测</u>，常规NMS很容易出现IoU过大而被抑制，漏检很大，斜向NMS就不会。  

### 2.3 Training objective (Multi-task loss)
![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/4.png)
* 细节如函数具体形式之类的就不说了

* RPN就是个二分类就行

* proposal的loss如上图公式。分为三部分：分类、正框reg、斜框reg。可见同时回归了正框和斜框的位置这个看似多余效果还行。



## 3. Experiments
![这里随便写文字](https://github.com/clw5180/CV_Paper/raw/master/res/R2CNN/5.png)

&emsp;&emsp;一张表就够了，可以看出：    
1. 加入正框预测的效果涨点值得一看，应该是让他更好学习    

2. 加入小尺度anchor效果不错（加一个大尺度anchor的效果劣化）    

3. 多尺度pooling的效果也还行，不是特别明显    

4. 斜框的NMS效果也还行，比3大比12小     


## Conclusion
* 采用多尺度pooling提取不同宽高比的信息
* 斜框的NMS解决传统NMS的密集漏检问题
* 正斜框的MTL（RRPN的思路不同）
