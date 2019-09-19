# Some Improvements on Deep Convolutional Neural Network Based Image Classification 

作者：Andrew G. Howard 

论文地址：https://arxiv.org/abs/1312.5402



## 一、介绍

TTA的鼻祖，后来查了下google scholar，发现本文不仅Inception V1有引用，许多其他重磅文章也都引用了该文，包括不限于VGG，He初始化，SSD等等。

该论文被引次数：截止2019.1.30，174次。次数虽然不多，不过引用该文的文章分量都很重。

2013年12月挂在arXiv上，属于AlexNet之后，VGG之前。由Andrew G. Howard一人发表，作者当时还是在一个以自己名字命名的咨询公司，不过现在已经去Google了，并在17年以一作身份发了MobileNets。这哥们也是神奇，这么多年总共就发了这两篇文章。

自注：个人感觉作者这里主要讲了几种能提高基于CNN的图像分类模型准确率的方法，包括：训练时的trick（如改进的crop策略、色彩增强），测试时trick（如多尺度）以及一些其他的细节，分别占据了论文的第2、3、4章节。



## 二、主要内容

- #### 训练时trick

  这里主要提了之前crop策略的改进和色彩方面的改进。

  （1）之前的crop策略为：之前是从256x256的图像中crop出224x224的图，但是256x256的图是通过将**长边**rescale到256，这样短边肯定小于256，这样如果仍然把短边扩大到256，相当于有很大一部分的无效区域；作者也提到了“This results in a loss of information by not considering roughly 30% of the pixels. While the cropped pixels are most likely less informative than the middle pixels we found that making use of these additional pixels improved the model. ” 

  对此，作者把**短边**缩放到256，这样长边就会大于256，就会得到N个256x256张图，N大于等于1。这样再从中crop出224x224的图，就无形之中增加了很多图片，扩充了训练集，模型会更好地学习到平移不变性。

  （2）色彩增强方面，之前只是增加了随机的亮度噪声，还增加了比如对比度、亮度和颜色方面的变化，用到了python image library（PIL）库。能够让神经网络学习到这些色彩方面的不变性。这里随机设置三种操作的顺序，并且取值在0.5到1.5之间变化，1代表不变；在这之后再添加随机噪声。

  

- #### 预测时trick

之前都是对10张图片进行预测然后求平均得到最终结果，主要是通过中心和四角的裁剪，以及它们的水平翻转来得到的这10张图片。但作者发现，在三个不同尺度下进行预测，也可以提高预测的效果。并且我们在三个不同的视角（就是不同的图片位置，类似于最早期的crop方式，只能看到完整图像的一部分）下进行预测。这样就包括了5个位置的crop，2种flip，3个scale以及3个视角（这里的意思是，对于上面rescale得到的256xN的图，选取3个位置的crop，得到了3个view，再在这3个view上面进行各种增强操作），一共90张图的预测，这样导致预测速度极慢，因此作者提出了一种贪心算法，如果从中选择10张图进行预测可以达到和90张图一样的效果，如果从中选择15张图甚至会优于90张图一些。

对于**多尺度预测，原始图像为256x256，作者这里额外增加了228和284两种尺度（自注：我这里计算了一下，就是原图尺寸0.9~1.1的一个范围）**，即进行3个scale的预测。作者提到，在图片缩放时，一定要选择一个好的差值方法，如双三次（bicubic）插值，and not to use anti aliasing filters designed for scaling images down.  



- #### 高分辨率模型

图片中的物体有不同的尺度，在前面我们在3种尺度下进行训练，但如果输入图片本身就是分辨率很高的，比如448x448，再用这3种尺度训练出来的网络进行预测就没有那么好了。实际上，这3种尺度下训练的网络可以用来初始化higher resolution model，可以**将训练时间从90epoch缩短到30epoch**。higher resolution model和基础模型是互补的，**1个基础网络和1个higher resolution model 融合的效果，和5个基础网络融合的效果相近**。

**模型细节**：

之前的模型是从256xN（Nx256）中取的224x224，理论上为了更高分辨率的模型，需要从448xN（Nx448）上取224x224，但实际上不可能这么存图，因此重用了256xN（Nx256）的图，从中采取128x128，然后再缩放到224x224。另外在高分辨率模型下，这些crop会重叠的更少，因此又增加了中上，中下，左中，右中4种crop。这样预测数量总计162个（9个crop，2个翻转，3个尺度和3个视角）



## 三、实验结果

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/TTA/1.png)

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/TTA/2.png)



## 四、结论

这个并不是TTA的鼻祖，TTA其实早在AlexNet时就用了，不过本文提出的数据增强方法比AlexNet要更有效。属于那种认真分析了问题，然后提出解决方案的文章。看似没什么多大的创新，但是非常有效。



主要参考：https://www.jianshu.com/p/27bac80d9fda