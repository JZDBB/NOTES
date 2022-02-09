# OCR basis

## OCR处理流程

在深度学习出现之前，传统算法(如积分投影、腐蚀膨胀、旋转等)在OCR领域占据主导地位。其标准的处理流程包括：图像预处理、文本行检测、单字符分割、单字符识别、后处理。

![img](./imgs/v2-093bfacaee568618f174bbb295509c20_1440w.png)

其中：

- 图像预处理主要是对图像的成像问题进行修正，包括几何变换（透视、扭曲、旋转等），去模糊、光线矫正等；
- 文本检测通常使用连通域、滑动窗口两个方向；
- 字符识别算法主要包括图像分类、模版匹配

深度学习OCR主要分为2步，首先是检测出图像中的文本行、接着进行文本识别。其中可以增加一步方向检测判断文字方向。

### 1. OCR 检测模型

文本检测就是要定位图像中的文字区域，然后通常以边界框的形式将单词或文本行标记出来。传统的文字检测算法多是通过手工提取特征的方式，特点是速度快，简单场景效果好，但是面对自然场景，效果会大打折扣。当前多是采用深度学习方法来做。

基于深度学习的文本检测算法可以大致分为以下几类：

1. 基于目标检测的方法；一般是预测得到文本框后，通过NMS筛选得到最终文本框，多是四点文本框，对弯曲文本场景效果不理想。典型算法为EAST、Text Box等方法。
2. 基于分割的方法；将文本行当成分割目标，然后通过分割结果构建外接文本框，可以处理弯曲文本，对于文本交叉场景问题效果不理想。典型算法为DB、PSENet等方法。
3. 混合目标检测和分割的方法；

### 2. OCR 识别模型

OCR识别算法的输入数据一般是文本行，背景信息不多，文字占据主要部分，识别算法目前可以分为两类算法：

1. 基于CTC的方法；即识别算法的文字预测模块是基于CTC的，常用的算法组合为CNN+RNN+CTC。目前也有一些算法尝试在网络中加入transformer模块等等。
2. 基于Attention的方法；即识别算法的文字预测模块是基于Attention的，常用算法组合是CNN+RNN+Attention。

### 3. PP-OCR模型

PaddleOCR 中集成了很多OCR算法，文本检测算法有DB、EAST、SAST等等，文本识别算法有CRNN、RARE、StarNet、Rosetta、SRN等算法。其中PaddleOCR针对中英文自然场景通用OCR，推出了PP-OCR系列模型，PP-OCR模型由DB+CRNN算法组成，利用海量中文数据训练加上模型调优方法，在中文场景上具备较高的文本检测识别能力。并且PaddleOCR推出了高精度超轻量PP-OCRv2模型，检测模型仅3M，识别模型仅8.5M，利用[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)的模型量化方法，可以在保持精度不降低的情况下，将检测模型压缩到0.8M，识别压缩到3M，更加适用于移动端部署场景。

## OCR实现效果

### 1. 文字检测

DB([paper]( https://arxiv.org/abs/1911.08947)) ，EAST([paper](https://arxiv.org/abs/1704.03155))，SAST([paper](https://arxiv.org/abs/1908.05498))。在ICDAR2015文本检测公开数据集上，算法效果如下：

| 模型 | 骨干网络    | precision | recall | Hmean  |                         
| ---- | ----------- | --------- | ------ | ------ |
| EAST | ResNet50_vd | 85.80%    | 86.71% | 86.25% |
| EAST | MobileNetV3 | 79.42%    | 80.64% | 80.03% | 
| DB   | ResNet50_vd | 86.41%    | 78.72% | 82.38% | 
| DB   | MobileNetV3 | 77.29%    | 73.08% | 75.12% | 
| SAST | ResNet50_vd | 91.39%    | 83.77% | 87.42% | 

在Total-text文本检测公开数据集上，算法效果如下：

| 模型 | 骨干网络    | precision | recall | Hmean  |
| ---- | ----------- | --------- | ------ | ------ |
| SAST | ResNet50_vd | 89.63%    | 78.44% | 83.66% | 

**说明：** SAST模型训练额外加入了icdar2013、icdar2017、COCO-Text、ArT等公开数据集进行调优。PaddleOCR用到的经过整理格式的英文公开数据集下载：

* [百度云地址](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (提取码: 2bpi)
* [Google Drive下载地址](https://drive.google.com/drive/folders/1ll2-XEVyCQLpJjawLDiRlvo_i4BqHCJe?usp=sharing)

### 2. 文本识别算法

CRNN([paper](https://arxiv.org/abs/1507.05717))、Rosetta([paper](https://arxiv.org/abs/1910.05085))、STAR-Net([paper](http://www.bmva.org/bmvc/2016/papers/paper043/index.html))、RARE([paper](https://arxiv.org/abs/1603.03915v1))、SRN([paper](https://arxiv.org/abs/2003.12294))、NRTR([paper](https://arxiv.org/abs/1806.00926v2))。参考[DTRB][3](https://arxiv.org/abs/1904.01906)文字识别训练和评估流程，使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法效果如下：

| 模型    | 骨干网络        | Avg Accuracy | 模型存储命名               |
| ------- | --------------- | ------------ | -------------------------- |
| Rosetta | Resnet34_vd     | 80.9%        | rec_r34_vd_none_none_ctc   | 
| Rosetta | MobileNetV3     | 78.05%       | rec_mv3_none_none_ctc      | 
| CRNN    | Resnet34_vd     | 82.76%       | rec_r34_vd_none_bilstm_ctc | 
| CRNN    | MobileNetV3     | 79.97%       | rec_mv3_none_bilstm_ctc    | 
| StarNet | Resnet34_vd     | 84.44%       | rec_r34_vd_tps_bilstm_ctc  |
| StarNet | MobileNetV3     | 81.42%       | rec_mv3_tps_bilstm_ctc     |
| RARE    | MobileNetV3     | 82.5%        | rec_mv3_tps_bilstm_att     | 
| RARE    | Resnet34_vd     | 83.6%        | rec_r34_vd_tps_bilstm_att  |
| SRN     | Resnet50_vd_fpn | 88.52%       | rec_r50fpn_vd_none_srn     | 
| NRTR    | NRTR_MTB        | 84.3%        | rec_mtb_nrtr               | 
