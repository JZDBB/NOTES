# 评估方案
- 准确率 （Accuracy） 、错误率 （Error rate）

- 混淆矩阵 （Confusion Matrix）

- 召回率（Recall）、精确率（Precision）

- P-R曲线、平均精度（Average-Precision，AP）、F指标

- 受试者工作特征曲线（Receiver Operating Characteristic，ROC）、AUC（Area Under Curve）、EER（Equal Error Rate）

- 平均精度均值(Mean Average Precision，mAP)、IOU(Intersection Over Union)



##### F指标 （重点）

综合精确度和召回率就是F指标：
$$
F_{\beta}=(1+{\beta}^2) \frac{{precision} · {recall}}{\beta^2 · precision +recall}
$$
$\beta$是关于召回的权重，大于1表示更看重召回，小于1表示看着精度，等于1就是F1-Measure。

##### ROC，AUC，EER
ROC关注两个指标：

1） True Positive Rate ( TPR ) = TP / [ TP + FN] ，TPR代表能将正例分对的概率

2）False Positive Rate( FPR ) = FP / [ FP + TN] ，FPR代表将负例错分为正例的概率

曲线左下和右上代表一个完全没有效果的分类器，如果曲线在对角线左上，说明分类器有效果，在右下说明是负效果。

越靠近左上效果越好，理想的分类器对应的ROC曲线和（0，0）、（0，1）、（1，1）所在折线重合。


AUC（Area Under Curve）：

ROC曲线围住的面积，越大，分类器效果越好。


EER（Equal Error Rate）：

指的是FNR=FPR的情况，因为FNR=1-TPR，所以在ROC曲线中就是曲线和（0，1）、（1，0）对角线的交点。从漏检和误检的角度，FPR理解为对正样本的漏检率，FNR则是预测为正样本的误检率。EER是均衡考虑这两者时的阈值选定标准。