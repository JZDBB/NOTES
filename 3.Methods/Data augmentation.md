# Data Augmentation

## image augmentation

通过对训练图像做一系列随机改变，来产生相似但又不同的训练样本，从而==扩大训练数据集==的规模。图像增广的另一种解释是，随机改变训练样本可以降低模型对某些属性的依赖，从而提高模型的==泛化能力==。

**普通的图像增强方法包括：翻转、旋转、平移、裁剪、缩放和高斯噪声；高级版图像增强方法还有常数填充、反射、边缘延伸、对称和包裹模式等**

### 常用的图像增广方法

- 翻转和裁剪

  左右翻转图像通常不改变物体的类别。它是最早也是最广泛使用的一种图像增广方法。上下翻转不如左右翻转通用。由于池化层能降低卷积层对目标位置的敏感度，因此我们还可以通过对图像随机裁剪来让物体以不同的比例出现在图像的不同位置，这同样能够降低模型对目标位置的敏感性。

- 变化颜色

  另一类增广方法是变化颜色。我们可以从四个方面改变图像的颜色：亮度、对比度、饱和度和色调。我们也可以随机变化图像的色调。

- 加噪声

- filter做锐化或者模糊

- 随机遮挡图片某一区域

### 叠加多个图像增广方法

实际应用中我们会将多个图像增广方法叠加使用。

### Multi-Crop

对于一个分类网络比如AlexNet，在测试阶段，使用single crop/multiple crop得到的结果是不一样的[0]，相当于将测试图像做数据增强。训练的时候随机剪裁，但测试的时候有技巧：

- 单纯将测试图像resize到某个尺度（例如256xN），选择其中centor crop（即图像正中间区域，比如224x224），作为CNN的输入，去评估该模型
- Multiple Crop的话具体形式有多种，可自行指定，比如：
  - 10个crops: 取（左上，左下，右上，右下，正中）以及它们的水平翻转。这10个crops在CNN下的预测输出取平均作为最终预测结果。
  - 144个crops：这个略复杂，以ImageNet为例：
    - 首先将图像resize到4个尺度（比如256xN，320xN，384xN，480xN）
    - 每个尺度上去取（最左，正中，最右）3个位置的正方形区域
    - 对每个正方形区域，取上述的10个224x224的crops，则得到4x3x10=120个crops
    - 对上述正方形区域直接resize到224x224，以及做水平翻转，则又得到4x3x2=24个crops
    - 总共加起来得到120+24=144个crops，所有crops的预测输出的平均作为整个模型对当前测试图像的输出

### 小结

* 图像增广基于现有训练数据生成随机图像从而应对过拟合。
* 为了在预测时得到确定的结果，我们通常只将图像增广应用在训练样本上，而不在预测时使用含随机操作的图像增广。



## 基于深度学习augmentation

- 特征空间增强（Feature Space Augmentation）：神经网络可以将图像这种高维向量映射为低维向量，在特征空间进行数据增强操作，例如：SMOTE算法，它是一种流行的增强方法，通过将k个最近的邻居合并以形成新实例来缓解类不平衡问题。
- 对抗生成（Adversarial Training）：对抗攻击表明，图像表示的健壮性远不及预期的健壮性。对抗生成可以改善学习的决策边界中的薄弱环节，提高模型的鲁棒性。
- 基于GAN的数据增强（GAN-based Data Augmentation）：使用 GAN 生成模型来生成更多的数据，可用作解决类别不平衡问题的过采样技术。
- 神经风格转换（Neural Style Transfer）：通过神经网络风格迁移来生成不同风格的数据，防止模型过拟合。