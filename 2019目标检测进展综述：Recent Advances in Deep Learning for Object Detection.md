# **2019目标检测进展综述：Recent Advances in Deep Learning for Object Detection**

论文地址：<https://arxiv.org/abs/1908.03673>



## 一、介绍

深度学习目标检测算法的里程碑：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/Overview/1.png)



## 二、主要内容

目标检测所涉及的主要内容、检测组件、学习策略、应用和基准测试：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/Overview/2.png)



著名的两阶段目标检测算法网络结构示意图：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/Overview/3.png)



著名的单阶段目标检测算法网络结构示意图：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/Overview/4.png)



特征表示部分多尺度学习的四种形式：分别为图像金字塔、预测金字塔、集成特征和特征金字塔。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/Overview/5.png)



目标检测的度量标准汇总：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/Overview/6.png)



## 三、实验结果

著名目标检测算法在PASCAL VOC数据集上的检测结果汇总：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/Overview/7.png)

自注：

1、ResNeXt50相比ResNet50提升约2个点，ResNeXt101相比ResNet101提升0.8个点，都是针对32x4d而言；如果用64x4d，效果还能好一点。另外从Detectron的Model Zoo来看，X-101-64x4d-FPN比X-101-32x8d-FPN只高0.1~0.2个点，with GN对于R-101-FPN来说只高了0.1个点，X-101-64x4d-FPN-cascade和传统baseline相比，AP50只高了0.2个点，但是整体AP高出了4个点。**表现最好的还是 X-101-64x4d-FPN-cascade**

2、三个mAP最高的对应的论文分别如下：

[115] T. Kong, F. Sun, W. Huang, H. Liu, Deep feature pyramid reconfiguration for object detection, in: ECCV, 2018. 

[140] H. Xu, X. Lv, X. Wang, Z. Ren, R. Chellappa, Deep regionlets for object
detection, in: ECCV, 2018.

[132] B. Cheng, Y. Wei, H. Shi, R. Feris, J. Xiong, T. Huang, Revisiting rcnn:
On awakening the classification power of faster rcnn, in: ECCV, 2018. 



著名目标检测算法在MS COCO数据集上的检测结果汇总：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/Overview/8.png)

最高mAP的为TridentNet，对应的论文如下：

[239] Y. Li, Y. Chen, N. Wang, Z. Zhang, Scale-aware trident networks for
object detection, in: arXiv preprint arXiv:1901.01892, 2019. 

知乎讲解：https://www.zhihu.com/search?type=content&q=TridentNet

代码实现：https://github.com/TuSimple/simpledet/tree/master/models/tridentnet



## 四、未来展望

作者在近年趋势基础上对未来目标检测的发展方向进行了展望：

**（1）Scalable Proposal Generation Strategy 可扩展的候选区域生成策略**

尤其是anchor-free相关的算法是最近的热点。

**（2）Effective Encoding of Contextual Information 上下文信息的有效编码**

上下文信息对于理解视觉世界是非常重要的，但目前这方面的文献还比较匮乏。

**（3）Detection based on Auto Machine Learning(AutoML) 基于AutoML的检测算法**

这虽是非常耗GPU的一个方向，但新出的工作不少，也取得了很不错的效果。

**（4）Emerging Benchmarks for Object Detection 新的目标检测基准测试数据集**

MS COCO虽然被广泛应用，但其仅有80类。而新出的LVIS数据集含有1000+个类别，164000幅图像，总计220万高质量实例分割Mask，各类别目标数量差异也很大。

A. Gupta, P. Dollar, R. Girshick, Lvis: A dataset for large vocabulary instance segmentation, in: CVPR, 2019.

**（5）Low-shot Object Detection 少样本目标检测**

业界已经提出了一些算法，但还有很大改进空间。

**（6）Backbone Architecture for Detection Task 适用于目标检测的骨干网结构**

大部分SOTA检测算法使用分类的骨干网，仅有少量算法使用检测专用骨干网。

**（7）Other Research Issues 其他研究话题**

比如大批量学习、增量学习等。
