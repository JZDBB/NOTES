**训练 GAN 的常用策略**

上一节都是基于一些简单的数学或者经验的分析，但是根本原因目前没有一个很好的理论来解释；尽管理论上的缺陷，我们仍然可以从一些经验中发现一些实用的
tricks，让你的 GANs 不再难训。这里列举的一些 tricks 可能跟 ganhacks
里面的有些重复，更多的是补充，但是为了完整起见，部分也添加在这里。

**1. model choice**

如果你不知道选择什么样的模型，那就选择 DCGAN[3] 或者 ResNet[4] 作为 base model。

**2. input layer**

假如你的输入是一张图片，将图片数值归一化到 [-1, 1]；假如你的输入是一个随机噪声的向量，最好是从 N(0, 1) 的正态分布里面采样，不要从
U(0,1) 的均匀分布里采样。

**3. output layer**

使用输出通道为 3 的卷积作为最后一层，可以采用 1x1 或者 3x3 的 filters，有的论文也使用 9x9 的
filters。（注：ganhacks 推荐使用 tanh）

**4. transposed convolution layer**

在做 decode 的时候，尽量使用 upsample+conv2d 组合代替 transposed_conv2d，可以减少 checkerboard
的产生 [5]；

在做超分辨率等任务上，可以采用 pixelshuffle [6]。在 tensorflow 里，可以用 tf.depth_to_sapce 来实现
pixelshuffle 操作。

**5. convolution layer**

由于笔者经常做图像修复方向相关的工作，推荐使用 gated-conv2d [7]。

**6. normalization**

虽然在 resnet 里的标配是 BN，在分类任务上表现很好，但是图像生成方面，推荐使用其他 normlization 方法，例如
parameterized 方法有 instance normalization [8]、layer normalization [9] 等，non-parameterized 方法推荐使用 pixel normalization [10]。假如你有选择困难症，那就选择大杂烩的 normalization
方法——switchable normalization [11]。

**7. discriminator**

想要生成更高清的图像，推荐 multi-stage discriminator
[10]。简单的做法就是对于输入图片，把它下采样（maxpooling）到不同 scale 的大小，输入三个不同参数但结构相同的
discriminator。

**8. minibatch discriminator**

由于判别器是单独处理每张图片，没有一个机制能告诉 discriminator 每张图片之间要尽可能的不相似，这样就会导致判别器会将所有图片都 push
到一个看起来真实的点，缺乏多样性。minibatch discriminator [22] 就是这样这个机制，显式地告诉 discriminator
每张图片应该要不相似。在 tensorflow 中，一种实现 minibatch discriminator 方式如下：

```python
 def minibatch_ discr iminator_ layer(x, output_ dim, hidden_ dim) :
# Improved Techniques for Training GANS(https://arxiv. org/pdf/1606. 03498. pdf)
      if x.ndim > 2:
      x = tf.layers.flatten(x)in_dim = shape_list(x)[-1]
      w = tf.get_variable("kernel",
                           [in_dim, hidden_dim, output_dim], tf.float32,
                           initializer=weight_initializer,
                           regularizer=weight_regularizer)
      xw = tf.einsum('ij, jkl->ikl', x, w)
      diff= tf.abs(tf.expand_dims (xw, axis=1), tf.expand_dims (xw, axis=0))
      o = tf.exp(-tf.reduce_sum(diff, axis=1))
      o = tf.reduce_sum(o, axis=1) # [batch, output_ dim]

	return O
```



上面是通过一个可学习的网络来显示度量每个样本之间的相似度，PGGAN
里提出了一个更廉价的不需要学习的版本，即通过统计每个样本特征每个像素点的标准差，然后取他们的平均，把这个平均值复制到与当前 feature map
一样空间大小单通道，作为一个额外的 feature maps 拼接到原来的 feature maps 里，一个简单的 tensorflow 实现如下：

```python
def minibatch_ discriminator_ layer(x) :23
"""
Progressive Growing of GANS for Improved Quality, Stability, and Variation(https://arxiv . org/pdf/ 1710.10196. pdf)
"""
	s = shape_list(x)
    adjusted_std = lambda x, **kwargs: tf.sqrt(
      		tf.reduce_mean((x - tf.reduce_ mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)
    vals= adjusted_std(input, axis=0, keep_dims=True)
    vals = tf.reduce_mean(vals, keep_dims=True)
    vals = tf.tile(vals, multiples=[s[0], s[l], s[2], 1])
    o = tf.concat([x, vals], axis=3)
    return o
```

**10. other loss**

- perceptual loss [17]
- style loss [18]
- total variation loss [17]
- l1 reconstruction loss

通常情况下，GAN loss 配合上面几种 loss，效果会更好。

**11. gradient penalty**

Gradient penalty 首次在 wgan-gp 里面提出来的，记为 1-gp，目的是为了让 discriminator 满足
1-lipchitchz 连续，后续 Mescheder, Lars M. et al [19] 又提出了只针对正样本或者负样本进行梯度惩罚，记为 0
-gp-sample。Thanh-Tung, Hoang et al [20] 提出了 0-gp，具有更好的训练稳定性。

**12. Spectral normalization [21]**

谱归一化是另外一个让判别器满足 1-lipchitchz 连续的利器，建议在判别器和生成器里同时使用。

ps: 在个人实践中，它比梯度惩罚更有效。

**13. one-size label smoothing [22]**

平滑正样本的 label，例如 label 1 变成 0.9-1.1 之间的随机数，保持负样本 label 仍然为 0。个人经验表明这个 trick
能够有效缓解训练不稳定的现象，但是不能根本解决问题，假如模型不够好的话，随着训练的进行，后期 loss 会飞。

**14. add supervised labels**

- add labels
- conditional batch normalization

**15. instance noise (decay over time)**

在原始 GAN 中，我们其实在优化两个分布的 JS 散度，前面的推理表明在两个分布的支撑集没有交集或者支撑集是低维的流形空间，他们之间的 JS
散度大概率上是 0；而加入 instance noise 就是强行让两个分布的支撑集之间产生交集，这样 JS 散度就不会为 0。新的 JS 散度变为：

![img](https://img.chainnews.com/material/images/a5068cf752fe39dfbf757f61ffedc149.jpg)

**16. TTUR [23]**

在优化 G 的时候，我们默认是假定我们的 D 的判别能力是比当前的 G 的生成能力要好的，这样 D 才能指导 G 朝更好的方向学习。通常的做法是先更新 D
的参数一次或者多次，然后再更新 G 的参数，TTUR 提出了一个更简单的更新策略，即分别为 D 和 G 设置不同的学习率，让 D 收敛速度更快。

**17. training strategy**

- PGGAN [10]

PGGAN 是一个渐进式的训练技巧，因为要生成高清（eg,
1024x1024）的图片，直接从一个随机噪声生成这么高维度的数据是比较难的；既然没法一蹴而就，那就循序渐进，首先从简单的低纬度的开始生成，例如
4x4，然后 16x16，直至我们所需要的图片大小。在 PGGAN
里，首次实现了高清图片的生成，并且可以做到以假乱真，可见其威力。此外，由于我们大部分的操作都是在比较低的维度上进行的，训练速度也不比其他模型逊色多少。

- coarse-to-refine

coarse-to-refine 可以说是 PGGAN 的一个特例，它的做法就是先用一个简单的模型，加上一个 l1
loss，训练一个模糊的效果，然后再把这个模糊的照片送到后面的 refine 模型里，辅助对抗 loss 等其他
loss，训练一个更加清晰的效果。这个在图片生成里面广泛应用。

**18. Exponential Moving Average [24]**

EMA 主要是对历史的参数进行一个指数平滑，可以有效减少训练的抖动。强烈推荐！！！





- [1]. Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
- [2]. Arjovsky, Martín and Léon Bottou. “Towards Principled Methods for Training Generative Adversarial Networks.” CoRR abs/1701.04862 (2017): n. pag.
- [3]. Radford, Alec et al. “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.” CoRR abs/1511.06434 (2016): n. pag.
- [4]. He, Kaiming et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 770-778.
- [5]. https://distill.pub/2016/deconv-checkerboard/
- [6]. Shi, Wenzhe et al. “Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 1874-1883.
- [7]. Yu, Jiahui et al. “Free-Form Image Inpainting with Gated Convolution.” CoRRabs/1806.03589 (2018): n. pag.
- [8]. Ulyanov, Dmitry et al. “Instance Normalization: The Missing Ingredient for Fast Stylization.” CoRR abs/1607.08022 (2016): n. pag.
- [9]. Ba, Jimmy et al. “Layer Normalization.” CoRR abs/1607.06450 (2016): n. pag.
- [10]. Karras, Tero et al. “Progressive Growing of GANs for Improved Quality, Stability, and Variation.” CoRR abs/1710.10196 (2018): n. pag.
- [11]. Luo, Ping et al. “Differentiable Learning-to-Normalize via Switchable Normalization.” CoRRabs/1806.10779 (2018): n. pag.
- [12]. Arjovsky, Martín et al. “Wasserstein GAN.” CoRR abs/1701.07875 (2017): n. pag.
- [13]. Mao, Xudong, et al. "Least squares generative adversarial networks." Proceedings of the IEEE International Conference on Computer Vision. 2017.
- [14]. Zhang, Han, et al. "Self-attention generative adversarial networks." arXiv preprint arXiv:1805.08318 (2018).
- [15]. Brock, Andrew, Jeff Donahue, and Karen Simonyan. "Large scale gan training for high fidelity natural image synthesis." arXiv preprint arXiv:1809.11096 (2018).
- [16]. Gulrajani, Ishaan et al. “Improved Training of Wasserstein GANs.” NIPS (2017).
- [17]. Johnson, Justin et al. “Perceptual Losses for Real-Time Style Transfer and Super-Resolution.” ECCV (2016).
- [18]. Liu, Guilin et al. “Image Inpainting for Irregular Holes Using Partial Convolutions.” ECCV(2018).
- [19]. Mescheder, Lars M. et al. “Which Training Methods for GANs do actually Converge?” ICML(2018).
- [20]. Thanh-Tung, Hoang et al. “Improving Generalization and Stability of Generative Adversarial Networks.” CoRR abs/1902.03984 (2018): n. pag.
- [21]. Yoshida, Yuichi and Takeru Miyato. “Spectral Norm Regularization for Improving the Generalizability of Deep Learning.” CoRR abs/1705.10941 (2017): n. pag.
- [22]. Salimans, Tim et al. “Improved Techniques for Training GANs.” NIPS (2016).
- [23]. Heusel, Martin et al. “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.” NIPS (2017).
- [24]. Yazici, Yasin et al. “The Unusual Effectiveness of Averaging in GAN Training.” CoRRabs/1806.04498 (2018): n. pag.







### 4. 训练的技巧

训练的技巧主要来自[Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)。

##### 1. 对输入进行规范化

- 将输入规范化到 -1 和 1 之间
- G 的输出层采用`Tanh`激活函数

##### 2. 采用修正的损失函数

在原始 GAN 论文中，损失函数 G 是要 min(log(1−D))min(log(1−D)), 但实际使用的时候是采用 max(logD)max(logD)，作者给出的原因是前者会导致梯度消失问题。

但实际上，即便是作者提出的这种实际应用的损失函数也是存在问题，即模式奔溃的问题，在接下来提出的 GAN 相关的论文中，就有不少论文是针对这个问题进行改进的，如 WGAN 模型就提出一种新的损失函数。

##### 3. 从球体上采样噪声

- 不要采用均匀分布来采样
- 从高斯分布中采样得到随机噪声
- 当进行插值操作的时候，从大圆进行该操作，而不要直接从点 A 到 点 B 直线操作，如下图所示

[![img](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/sphere.png)](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/sphere.png)

- 更多细节可以参考 Tom White’s 的论文 [Sampling Generative Networks](https://arxiv.org/abs/1609.04468) 以及代码https://github.com/dribnet/plat

##### 4. BatchNorm

- 采用 mini-batch BatchNorm，要保证每个 mini-batch 都是同样的真实图片或者是生成图片
- 不采用 BatchNorm 的时候，可以采用 instance normalization（对每个样本的规范化操作）
- 可以使用**虚拟批量归一化**(virtural batch normalization):开始训练之前预定义一个 batch R，对每一个新的 batch X，都使用 R+X 的级联来计算归一化参数

##### 5. 避免稀疏的梯度：Relus、MaxPool

- 稀疏梯度会影响 GAN 的稳定性
- 在 G 和 D 中采用 LeakyReLU 代替 Relu 激活函数
- 对于下采样操作，可以采用平均池化(Average Pooling) 和 Conv2d+stride 的替代方案
- 对于上采样操作，可以使用 PixelShuffle(https://arxiv.org/abs/1609.05158), ConvTranspose2d + stride

##### 6. 标签的使用

- 标签平滑。也就是如果有两个目标标签，假设真实图片标签是 1，生成图片标签是 0，那么对每个输入例子，如果是真实图片，采用 0.7 到 1.2 之间的一个随机数字来作为标签，而不是 1；一般是采用单边标签平滑
- 在训练 D 的时候，偶尔翻转标签
- 有标签数据就尽量使用标签

##### 7. 使用 Adam 优化器

##### 8. 尽早追踪失败的原因

- D 的 loss 变成 0，那么这就是训练失败了
- 检查规范的梯度：如果超过 100，那出问题了
- 如果训练正常，那么 D loss 有低方差并且随着时间降低
- 如果 g loss 稳定下降，那么它是用糟糕的生成样本欺骗了 D

##### 9. 不要通过统计学来平衡 loss

##### 10. 给输入添加噪声

- 给 D 的输入添加人为的噪声
  - http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
  - https://openreview.net/forum?id=Hk4_qw5xe
- 给 G 的每层都添加高斯噪声

##### 11. 对于 Conditional GANs 的离散变量

- 使用一个 Embedding 层
- 对输入图片添加一个额外的通道
- 保持 embedding 低维并通过上采样操作来匹配图像的通道大小

##### 12 在 G 的训练和测试阶段使用 Dropouts

- 以 dropout 的形式提供噪声(50%的概率)
- 训练和测试阶段，在 G 的几层使用
- https://arxiv.org/pdf/1611.07004v1.pdf