# **YOLOv1：You Only Look Once Unified, Real Time Object Detection**

论文地址：<http://arxiv.org/abs/1506.02640>

代码复现：<https://github.com/abhi2610/ohem>

&emsp;&emsp;&emsp;&emsp;&emsp;https://github.com/pjreddie/darknet

&emsp;&emsp;&emsp;&emsp;&emsp;https://github.com/hizhangp/yolo_tensorflow （tensorflow版本）



## 一. 介绍

&emsp;&emsp;一阶段方法（End to End方法）主要有SSD系列，YOLO系列，这种方法是将目标边框的定位问题转化为回归问题处理。由于思想的不同，二阶段检测方法在检测准确率和定位精度有优势，一阶段检测方法在速度上占有优势。

&emsp;&emsp;所以YOLO的核心思想是，直接在输出层回归bounding box的位置和bounding box所属的类别（整张图作为网络的输入，把 Object Detection 的问题转化成一个 Regression 问题）。YOLOv1 算法省掉了生成候选框的步骤，直接用一个检测网络进行端到端的检测，因而检测速度非常快。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/0.png)





## 二. 主要内容

#### 1. 模型架构

&emsp;&emsp;网络架构如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/1.png)

&emsp;&emsp;设计 YOLOv1 的网络结构时参考了 GoogLeNet 的网络结构，但并未使用 Inception 的通道组合策略，而是大量使用了 1x1 和 3x3 卷积。前 24 层卷积用来提取图像特征，后面 2 层全连接用来预测目标位置和类别概率。在训练时先利用 ImageNet 分类数据集对前 20 层卷积层进行预训练，将预训练结果再加上剩下的四层卷积以及 2 层全连接，采用了 Leaky Relu 作为激活函数，其中为了防止过拟合对全连接层加了失活概率为 0.5 的 dropout 层。

&emsp;&emsp;从图中可以看到，yolo网络的输出的网格是7x7大小的（自注：7x7是对于448x448的输入，相当于下采样到原来的1/64），另外，输出的channel数目为30。一个cell内，前20个元素是类别概率值，然后2个元素是边界框confidence，最后8个元素是边界框的 (x, y, w, h) 。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/2_0.png)

&emsp;&emsp;也就是说，每个cell有两个predictor，每个predictor分别预测一个bounding box的x，y，w，h和相应的confidence。但分类部分的预测却是共享的。**因此同一个cell内是没办法预测多个目标的**。（注：如果一个cell要预测两个目标，那么这两个predictor要怎么分工预测这两个目标？谁负责谁？不知道，所以没办法预测。**而像faster rcnn这类算法，可以根据anchor与ground truth的IOU大小来安排anchor负责预测哪个物体，所以后来yolo2也采用了anchor思想，同个cell才能预测多个目标**）



#### 2. 预测步骤

YOLOv1的预测步骤如下图所示：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/5_0.png)

(1) 将图像划分成**S x S**的网格。因为网格的数目需要跟最后的特征图尺寸完全相同才能保证一一对应，所以对于448x448的输入，最后输出特征图大小为 7 x 7，所以S=7。至于**为什么要分成S x S个网格**，是因为用于预测的特征图大小为 7 x 7，即一共有49个向量（这里有点像RPN），每个向量都要去预测“框”和“类别”，那么训练的时候，我们需要为每个向量分配类别以及是否需要负责预测框，如何分配，我们需要把49个点映射回原图，正好形成7 x 7个网格，然后根据每个网格跟Ground Truth之间的关系，来做后续分配。**这里和RPN的不同之处**见下图：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/5_1.png)

**物体的“中心”落在哪个Grid Cell中，哪个Grid Cell就负责训练这个Ground Truth**。**这张图“橘黄色的Grid Cell”包括了Ground Truth中心点，那么它就是所谓的正例**，**尽管“绿色的Grid Cell”和“白色的Grid Cell”与Ground Truth IOU很大，也是负例**，因为这里有且只有一个Grid Cell是正例；所以白色和绿色的Cell也就不用负责训练框了。另外，**实际的训练和预测的过程中，我们是不需要对原图划分网格的，论文中划分网格的目的主要是为了方便表达**。


(2) 每个网格预测2个bounding box，总共输出7×7个长度为30的tensor

(3) 根据上一步可以预测出7 * 7 * 2 = 98个目标窗口，然后根据阈值去除可能性比较低的目标窗口，再由NMS去除冗余窗口即可。具体步骤见下图：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/5_2.png)

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/5_3.png)

YOLOv1使用了end-to-end的回归方法，没有region proposal步骤，直接回归便完成了位置和类别的判定。种种原因使得YOLOv1在目标定位上不那么精准，直接导致YOLO的检测精度并不是很高。



#### 3.模型输出的意义

**Confidence预测**

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/3.png)

**注意：实际上，在测试阶段，网络只是输出了confidence这个值**，它已经包含了 Pr(object) * IOU ，而不是用Pr(object) * IOU来计算的（以为没有gt，没办法算）。在训练阶段，置信度的衡量标准是 score = Pr(object) * IOU 这个值，**而在测试的时候，网络会直接吐出来score这个值**。  



**Bounding box预测**

bounding box的预测包括**x, y, w, h**四个值。**x和y表示bounding box的中心相对于当前所在cell的归一化坐标偏移，比如x=0.5,y=0.5就说明bbox中心和当前cell中心点重合，而x=0.6,y=0.6表示在当前cell中心点的右下方；值在0~1范围内**（自注：这里的“坐标”是以cell单元格在宽高方向上的个数为基准，比如row=1，col=4，而不是以图片实际实际宽高为基准）；w和h则是相对于整张图片的宽高进行归一化的。

**偏移的计算方法**如下图所示。

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/4.png)

**举例说明**：比如上图S=10即一共10 x 10个grid，原图的宽高均为70pixels，假如某个x在原图上的坐标是38.5，则有38.5 x 10 / 70 - 5 = 0.5，说明bbox的中心点位于当前cell的中心；简单证明一下，原图宽高为70x70，那么归一化之后原图坐标x=38.5的位置对应x=0.5，x=42归一化后对应x=0.6，则38.5这个位置恰好在中间，即x=0.55，为归一化后的x坐标，但这里又算的是相对当前cell左上角的偏移，因此x=(0.55-0.5) / (0.6-0.5) = 0.5，相当于偏离了一半，也就是在中间位置。 

**再举一例做详细说明**：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/4_0.png)

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/4_1.png)

**原因**：实际上经过这么表示之后，x，y，w，h都归一化了，它们的值都是在0 ~ 1之间。我们**通常做回归问题的时候都会将输出进行归一化，否则可能导致各个输出维度的取值范围差别很大，进而导致训练的时候，网络更关注数值大的维度。因为数值大的维度，算loss相应会比较大，为了让这个loss减小，那么网络就会尽量学习让这个维度loss变小，最终导致区别对待**。



**类别预测**

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/5.png)



#### 3、训练

loss如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/6.png)

将上述损失函数拆解如下：

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/6_0.png)

**三部分损失**的类型都为**误差平方和**，**不同是前面的权重系数，用来衡量三者各自的重要性**。

自注：论文中提到：“We set λcoord = 5 and λnoobj = 0.5”，说明两点：**1、**位置相关误差（坐标、IOU）与分类误差对网络loss的贡献值是不同的，因此YOLO在计算loss时，使用λcoord =5修正coordError。**2、**在计算IOU误差时，包含物体的格子与不包含物体的格子，二者的IOU误差对网络loss的贡献值是不同的，**主要惩罚的是包含物体的grid cells**，要使其损失接近于零；**“我们不需要一个能够判断哪些背景的检测器，而是需要一个能够发现哪些是object的检测器”**；还有一点，小的box的偏移带来的影响要超过大的box偏移，为强调这一点，预测的是box宽和高的平方根而不是其本身。

关于loss，需要特别注意的是需要计算loss的部分。并不是网络的输出都算loss，具体地说：

1. 有物体中心落入的cell的情况：1、预测的bounding box与ground truth IoU比较大的那个predictor需要计算x，y，w，h的loss；2、预测的bounding box与ground truth IoU比较大的那个predictor要计算confidence loss；3、两个predictor都需要计算分类loss；
2. 没有物体中心落入的cell的情况：只需要计算confidence loss。

另外，我们发现**每一项loss的计算都是L2 loss，即使是分类问题也是。所以说YOLO是把分类问题转为了回归问题**。



## 三. 实验结果

![这里随便写文字](https://github.com/clw5180/CV_Paper/blob/master/res/YOLOv1/7.png)



## 四. 结论

* YOLOv1优点：1.速度快  2.在检测物体时能很好的利用上下文信息，不容易在背景上预测出错误的物体信息
* 缺点：1.容易产生物体的定位错误。2.对小物体的检测效果不好（尤其是密集的小物体，因为同一cell只能预测2个物体，且不能预测同一类物体）。 3、由于YOLO中采用了全联接层，所以需要在检测时，读入测试的图像的大小必须和训练集的图像尺寸相同；



### 补充说明

yolo反向传播的时候如何计算梯度呢？

yolo最终的输出，实际是全连接层，reshape的结果只是方便与前面映射回去，所以在计算梯度的时候，对loss进行求导，在全连接层相应位置映射回去便可以了。
详情可参考代码如下：

forward_detection_layer函数中，delta为梯度，读者可执行对应其计算过程
make_detection_layer中有初始化过程，读者可以对应其值

```python
detection_layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
{
    detection_layer l = {0};
    l.type = DETECTION;

    l.n = n;   ##  及论文中 B = 2
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.side = side;    ###  即论文中 S = 7 
    l.w = side;
    l.h = side;
    assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs;
    l.truths = l.side*l.side*(1+l.coords+l.classes);
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));

    l.forward = forward_detection_layer;
    l.backward = backward_detection_layer;
#ifdef GPU
    l.forward_gpu = forward_detection_layer_gpu;
    l.backward_gpu = backward_detection_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return l;
}

void forward_detection_layer(const detection_layer l, network net)
{
    int locations = l.side*l.side;
    int i,j;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    //if(l.reorg) reorg(l.output, l.w*l.h, size*l.n, l.batch, 1);
    int b;
    if (l.softmax){
        for(b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int offset = i*l.classes;
                softmax(l.output + index + offset, l.classes, 1, 1,
                        l.output + index + offset);
            }
        }
    }
    if(net.train){
        float avg_iou = 0;
        float avg_cat = 0;
        float avg_allcat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int truth_index = (b*locations + i)*(1+l.coords+l.classes);
                int is_obj = net.truth[truth_index];
                for (j = 0; j < l.n; ++j) {
                    int p_index = index + locations*l.classes + i*l.n + j;
                    l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
                    *(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
                    avg_anyobj += l.output[p_index];
                }

                int best_index = -1;
                float best_iou = 0;
                float best_rmse = 20;

                if (!is_obj){
                    continue;
                }

                int class_index = index + i*l.classes;
                for(j = 0; j < l.classes; ++j) {
                    l.delta[class_index+j] = l.class_scale * (net.truth[truth_index+1+j] - l.output[class_index+j]);
                    *(l.cost) += l.class_scale * pow(net.truth[truth_index+1+j] - l.output[class_index+j], 2);
                    if(net.truth[truth_index + 1 + j]) avg_cat += l.output[class_index+j];
                    avg_allcat += l.output[class_index+j];
                }

                box truth = float_to_box(net.truth + truth_index + 1 + l.classes, 1);
                truth.x /= l.side;
                truth.y /= l.side;

                for(j = 0; j < l.n; ++j){
                    int box_index = index + locations*(l.classes + l.n) + (i*l.n + j) * l.coords;
                    box out = float_to_box(l.output + box_index, 1);
                    out.x /= l.side;
                    out.y /= l.side;

                    if (l.sqrt){
                        out.w = out.w*out.w;
                        out.h = out.h*out.h;
                    }

                    float iou  = box_iou(out, truth);
                    //iou = 0;
                    float rmse = box_rmse(out, truth);
                    if(best_iou > 0 || iou > 0){
                        if(iou > best_iou){
                            best_iou = iou;
                            best_index = j;
                        }
                    }else{
                        if(rmse < best_rmse){
                            best_rmse = rmse;
                            best_index = j;
                        }
                    }
                }

                if(l.forced){
                    if(truth.w*truth.h < .1){
                        best_index = 1;
                    }else{
                        best_index = 0;
                    }
                }
                if(l.random && *(net.seen) < 64000){
                    best_index = rand()%l.n;
                }

                int box_index = index + locations*(l.classes + l.n) + (i*l.n + best_index) * l.coords;
                int tbox_index = truth_index + 1 + l.classes;

                box out = float_to_box(l.output + box_index, 1);
                out.x /= l.side;
                out.y /= l.side;
                if (l.sqrt) {
                    out.w = out.w*out.w;
                    out.h = out.h*out.h;
                }
                float iou  = box_iou(out, truth);

                //printf("%d,", best_index);
                int p_index = index + locations*l.classes + i*l.n + best_index;
                *(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
                *(l.cost) += l.object_scale * pow(1-l.output[p_index], 2);
                avg_obj += l.output[p_index];
                l.delta[p_index] = l.object_scale * (1.-l.output[p_index]);

                if(l.rescore){
                    l.delta[p_index] = l.object_scale * (iou - l.output[p_index]);
                }

                l.delta[box_index+0] = l.coord_scale*(net.truth[tbox_index + 0] - l.output[box_index + 0]);
                l.delta[box_index+1] = l.coord_scale*(net.truth[tbox_index + 1] - l.output[box_index + 1]);
                l.delta[box_index+2] = l.coord_scale*(net.truth[tbox_index + 2] - l.output[box_index + 2]);
                l.delta[box_index+3] = l.coord_scale*(net.truth[tbox_index + 3] - l.output[box_index + 3]);
                if(l.sqrt){
                    l.delta[box_index+2] = l.coord_scale*(sqrt(net.truth[tbox_index + 2]) - l.output[box_index + 2]);
                    l.delta[box_index+3] = l.coord_scale*(sqrt(net.truth[tbox_index + 3]) - l.output[box_index + 3]);
                }

                *(l.cost) += pow(1-iou, 2);
                avg_iou += iou;
                ++count;
            }
        }

        if(0){
            float *costs = calloc(l.batch*locations*l.n, sizeof(float));
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        costs[b*locations*l.n + i*l.n + j] = l.delta[p_index]*l.delta[p_index];
                    }
                }
            }
            int indexes[100];
            top_k(costs, l.batch*locations*l.n, 100, indexes);
            float cutoff = costs[indexes[99]];
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        if (l.delta[p_index]*l.delta[p_index] < cutoff) l.delta[p_index] = 0;
                    }
                }
            }
            free(costs);
        }


        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);


        printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(count*l.classes), avg_obj/count, avg_anyobj/(l.batch*locations*l.n), count);
        //if(l.reorg) reorg(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    }
}
```


参考：<https://blog.csdn.net/zsl091125/article/details/84953166>

<https://blog.csdn.net/Chunfengyanyulove/article/details/80566601> （反向传播部分）
