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





## 一、权重

**a. 调节Generator loss中GAN loss的权重**
G loss和Gan loss在一个尺度上或者G loss比Gan loss大一个尺度。但是千万不能让Gan loss占主导地位, 这样整个网络权重会被带偏。

## 二、训练次数

**b. 调节Generator和Discrimnator的训练次数比**
一般来说，Discrimnator要训练的比Genenrator多。比如训练五次Discrimnator，再训练一次Genenrator(WGAN论文 是这么干的)。

## 三、学习率

**c. 调节learning rate**
这个学习速率不能过大。一般要比Genenrator的速率小一点。

## 四、优化器

**d. Optimizer的选择不能用基于动量法的**
如Adam和momentum。可使用RMSProp或者SGD。

## 五、结构

**e. Discrimnator的结构可以改变**
如果用WGAN，判别器的最后一层需要去掉sigmoid。但是用原始的GAN，需要用sigmoid，因为其loss function里面需要取log，所以值必须在[0,1]。这里用的是邓炜的critic模型当作判别器。之前twitter的论文里面的判别器即使去掉了sigmoid也不好训练。

## 六、 loss曲线

**f. Generator loss的误差曲线走向**
因为Generator的loss定义为： G_loss = -tf.reduce_mean(D_fake) Generator_loss = gen_loss + lamda*G_loss	其中gen_loss为Generator的loss，G_loss为Discrimnator的loss，目标是使Generator_loss不断变小。所以理想的Generator loss的误差曲线应该是不断往0靠的下降的抛物线。
**g. Discrimnator loss的误差曲线走向**
因为Discrimnator的loss定义为： D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)这个是一个和Generator抗衡的loss。目标就是使判别器分不清哪个是生成器的输出哪个是真实的label。所以理想的Discrimnator loss的误差曲线应该是最终在0附近振荡，即傻傻分不清。换言之，就是判别器有50%的概率判断你是真的，50%概率判断你是假的。











使用 GAN 的缺点 



GAN 难以使用的原因有很多，这里列出一些主要的原因。



**1、模式坍塌 (Mode collapse)**



自然数据分布是高度复杂且多模态的。也就是说，数据分布有很多 **“峰值”(peaks)** 或 **“模式”(modes)**。每个 mode 表示相似数据样本的集中度，但与其他 mode 不同。



在 mode collapse 期间，生成器生成属于**一组有限模式集**的样本。当生成器认为它可以通过锁定单个模式来欺骗鉴别器时，就会发生这种情况。也就是说，生成器仅从这种模式来生成样本。



![img](http://cdn.zhuanzhi.ai/images/wx/43445ec5156cf446e1133fb7611353c0)

顶部的图像表示没有发生 mode collapse 的 GAN 的输出。底部的图像表示发生了 mode collapse 的 GAN 的输出

(Source: https://arxiv.org/pdf/1611.02163.pdf)



鉴别器最终会**发现****这种模式下的样本是假的**。但生成器仅仅是锁定到另一种模式。这个循环无限重复，**从根本上限制了生成样本的多样性**。



**2、收敛 (Convergence)**



GAN 训练中一个常见的问题是 “**我们应该在什么时候停止训练？**”。由于鉴别器损失降低时，生成器损失增加 (反之亦然)，我们不能根据损失函数的值来判断收敛性。如下图所示:



![img](http://cdn.zhuanzhi.ai/images/wx/7e3a7e59cdc8dcf7fb82e4c810f8234c)

一个典型的 GAN 损失函数



**3. 质量**



与前一个问题一样，很难定量地判断生成器何时产生高质量的样品。在损失函数中加入额外的感知正则化可以在一定程度上缓解这种情况。



**4. 度量标准 (Metrics)**



GAN 目标函数可以解释生成器或鉴别器相对于其他方法的性能表现。然而，它并不代表输出的质量或多样性。因此，我们需要不同的度量标准。



8大技巧提高GAN性能 



有很多技巧可以用来使 GAN 更加稳定或更加强大。这里只解释了相对较新的或较复杂的一些技术。



**1、替代损失函数 (Alternative Loss Functions）**



针对 GAN 的缺陷，最常用的一种修复方法是 **Wasserstein GAN**。它本质上用 **Earth Mover distance** (Wasserstein-1 distance 或 EM distance) 来替代传统 GAN 的 **Jensen Shannon 散度**。EM 距离的原始形式是难以处理的，因此我们使用它的 dual 形式。这要求鉴别器为 1-Lipschitz，它是通过削减鉴别器的权重来维持的。



使用 Earth Mover distance 的优点是，即使真实的数据和生成的数据分布不相交，它也是**连续的**，这与 JS 散度或 KL 散度不同。同时，生成的图像质量与损失值之间存在相关性。缺点是，我们需要对每个生成器更新执行多个鉴别器更新。此外，作者认为，利用权重削减来确保 1-Lipschitz 约束是一种糟糕的方法。



![img](http://cdn.zhuanzhi.ai/images/wx/e2eff6fdc2ebf14e3e97f5b5c2269e11)

即使分布不连续，earth mover distance(左)也是连续的，与 JS 散度 (右) 不同



另一个解决方案是使用**均方损失 (mean squared loss)** 来替代**对数损失**。LSGAN 的作者认为，传统的 GAN 损失函数并没有提供太多的激励来将生成的数据分布 “拉” 到接近真实数据分布的位置。



原始 GAN 损失函数中的 log loss 并不关心生成的数据与决策边界的距离 (决策边界将真实数据和虚假数据分开)。另一方面，LSGAN 对远离决策边界的生产样本实施乘法，本质上是将生成的数据分布 “**拉**” 得更接近真实的数据分布。LSGAN 用均方损失代替对数损失来实现这一点。



**2、Two Timescale Update Rule (TTUR)**



在这种方法中，我们对鉴别器和生成器使用不同的学习率。通常，生成器使用较慢的更新规则 (update rule)，鉴别器使用较快的更新规则。使用这种方法，我们可以以 1:1 的比例执行生成器和识别器的更新，只需要修改学习率。SAGAN 实现正是使用了这种方法。



**3、梯度惩罚 (Gradient Penalty)**



在 Improved Training of WGANs 这篇论文中，作者声称 **weight clipping** 会导致优化问题。



作者表示， weight clipping 迫使神经网络学习最优数据分布的 “更简单的近似”，从而导致较低质量的结果。他们还声称，如果没有正确设置 WGAN 超参数，那么 weight clipping 会导致梯度爆炸或梯度消失问题。



作者在损失函数中引入了一个简单的 **gradient penalty**，从而缓解了上述问题。此外，与最初的 WGAN 实现一样，保留了 1-Lipschitz 连续性。



![img](http://cdn.zhuanzhi.ai/images/wx/ba6bebf0333e07641036da9dec1bb992)

与 WGAN-GP 原始论文一样，添加了 gradient penalty 作为一个正则化器



DRAGAN 的作者声称，当 GAN 所玩的游戏达到 “局部平衡状态” 时，就会发生 mode collapse。他们还声称，鉴别器围绕这些状态产生的梯度是“尖锐的”。当然，使用 gradient penalty 可以帮助我们避开这些状态，大大增强稳定性，减少模式崩溃。



**4、谱归一化 (Spectral Normalization)**



Spectral Normalization 是一种**权重归一化技术**，通常用于鉴别器上，以增强训练过程。这本质上保证了鉴别器是 **K-Lipschitz** 连续的。



像 SAGAN 这样的一些实现，也在生成器上使用 spectral Normalization。该方法比梯度惩罚法计算效率更高。



**5、Unrolling 和 Packing**



防止 mode hopping 的一种方法是预测未来，并在更新参数时预测对手。Unrolled GAN 使生成器能够在鉴别器有机会响应之后欺骗鉴别器。



防止 mode collapse 的另一种方法是在将属于同一类的多个样本传递给鉴别器之前 “打包” 它们，即 **packing**。这种方法被 PacGAN 采用，在 PacGAN 论文中，作者报告了 mode collapse 有适当减少。



**6、堆叠 GAN**



单个 GAN 可能不足以有效地处理任务。我们可以使用多个连续堆叠的 GAN，其中每个 GAN 可以解决问题中更简单的一部分。例如，FashionGAN 使用两个 GAN 来执行局部图像翻译。



![img](http://cdn.zhuanzhi.ai/images/wx/8eb636fe9c56b054bb4884af48b71c75)

FashionGAN 使用两个 GAN 进行局部图像翻译



把这个概念发挥到极致，我们可以逐渐加大 GAN 所解决的问题的难度。例如， **Progressive GAN (ProGAN)** 可以生成高质量的高分辨率图像。



**7、Relativistic GAN**



传统的 GAN 测量生成的数据是真实数据的概率。 Relativistic GAN 测量生成的数据比真实数据 “更真实” 的概率。正如 RGAN 论文中提到的，我们可以使用适当的距离度量来度量这种“相对真实性”。



![img](http://cdn.zhuanzhi.ai/images/wx/5739c5e9cfced98548a6ab0e2a06a83d)

使用标准 GAN loss 时鉴别器的输出 (图 B)。图 C 表示输出曲线的实际样子。图 A 表示 JS 散度的最优解。



作者还提到，鉴别器的输出在达到最优状态时应该收敛到 0.5。然而，传统的 GAN 训练算法强迫鉴别器对任何图像输出 “real”(即 1)。这在某种程度上阻止了鉴别器达到其最优值。 relativistic 方法也解决了这个问题，并取得了相当显著的效果，如下图所示：



![img](http://cdn.zhuanzhi.ai/images/wx/d9f148b60b3e0be5b81cb2e38f86b536)

经过 5000 次迭代后，标准 GAN(左) 和 relativistic GAN(右) 的输出



**8、自注意力机制**



Self Attention GANs 的作者表示，用于生成图像的卷积会查看局部传播的信息。也就是说，由于它们限制性的 receptive field，它们错过了全局性的关系。



![img](http://cdn.zhuanzhi.ai/images/wx/832c57e5f5ef19bff53a9e75f45dbd6a)

将 attention map(在黄色框中计算) 添加到标准卷积操作中



Self-Attention GAN 允许对图像生成任务进行注意力驱动的长期依赖建模。 Self-Attention 机制是对普通卷积运算的补充。全局信息 (远程依赖) 有助于生成更高质量的图像。网络可以选择忽略注意机制，也可以将其与正常卷积一起考虑。



![img](http://cdn.zhuanzhi.ai/images/wx/1795054c7e2aeccb351e34dc2ad1afe0)

对红点标记的位置的 attention map 的可视化



总结



研究社区已经提出了许多解决方案和技巧来克服 GAN 训练的缺点。然而，由于新研究的数量庞大，很难跟踪所有重要的贡献。



由于同样的原因，这篇文章中分享的细节并非详尽无疑，可能在不久的将来就会过时。尽管如此，还是希望本文能够成为人们寻找改进 GAN 性能的方法的一个指南。











训练WGAN的时候，有几个方面可以调参：

   a. 调节Generator loss中GAN loss的权重。 G loss和Gan loss在一个尺度上或者G loss比Gan loss大一个尺度。但是千万不能让Gan loss占主导地位, 这样整个网络权重会被带偏。

   b. 调节Generator和Discrimnator的训练次数比。一般来说，Discrimnator要训练的比Genenrator多。比如训练五次Discrimnator，再训练一次Genenrator(WGAN论文 是这么干的)。

   c. 调节learning rate，这个学习速率不能过大。一般要比Genenrator的速率小一点。

   d. Optimizer的选择不能用基于动量法的，如Adam和momentum。可使用RMSProp或者SGD。

   e. Discrimnator的结构可以改变。如果用WGAN，判别器的最后一层需要去掉sigmoid。但是用原始的GAN，需要用sigmoid，因为其loss function里面需要取log，所以值必须在[0,1]。这里用的是邓炜的critic模型当作判别器。之前twitter的论文里面的判别器即使去掉了sigmoid也不好训练。

   f. Generator loss的误差曲线走向。因为Generator的loss定义为：

  	G_loss = -tf.reduce_mean(D_fake)

​    	Generator_loss = gen_loss + lamda*G_loss

​	其中gen_loss为Generator的loss，G_loss为Discrimnator的loss，目标是使Generator_loss不断变小。所以理想的Generator loss的误差曲线应该是不断往0靠的下降的抛物线。

   g. Discrimnator loss的误差曲线走向。因为Discrimnator的loss定义为：

​      D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)

这个是一个和Generator抗衡的loss。目标就是使判别器分不清哪个是生成器的输出哪个是真实的label。所以理想的Discrimnator loss的误差曲线应该是最终在0附近振荡，即傻傻分不清。换言之，就是判别器有50%的概率判断你是真的，50%概率判断你是假的。

   h. 之前的想法是就算判别器不训练，那么它判断这个图片是真是假的概率都是50%，那D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)不就已经在0附近了吗？

其实不是这样的。如果是wgan的话，判别器的输出是一个负无穷到正无穷的数值，那么要让它对两个不同的输入产生相似的输出是很难的。同理，对于gan的话，判别器的输出是介于[0,1]之间的，产生两个相似的输出也是很困难的。如果判别器的输出是0或者1的话，那就是上面说的情况。所以，网络要经过学习，使得 输出尽可能相似，那就达到了傻傻分不清的状态了。













尽管 GAN 领域的进步令人印象深刻，但其在应用过程中仍然存在一些困难。本文梳理了 GAN 在应用过程中存在的一些难题，并提出了最新的解决方法。

 

### 

 

#### 使用 GAN 的缺陷

 

众所周知，GAN 是由 Generator 生成网络和 Discriminator 判别网络组成的。

 

\1. Mode collapse（模型奔溃）

 

注：Mode collapse 是指 GAN 生成的样本单一，其认为满足某一分布的结果为 true，其他为 False，导致以上结果。

 

自然数据分布是非常复杂，且是多峰值的（multimodal）。也就是说数据分布有很多的峰值（peak）或众数（mode）。每个 mode 都表示相似数据样本的聚集，但与其他 mode 是不同的。

 

在 mode collapse 过程中，生成网络 G 会生成属于有限集 mode 的样本。当 G 认为可以在单个 mode 上欺骗判别网络 D 时，G 就会生成该 mode 外的样本。

 

![img](http://flashgene.com/wp-content/uploads/2019/02/51d6e8c94839474d162acb1ba7f1e280.jpg)

 

上图表示 GAN 的输出没有 mode collapse. 下图则出现了 mode collapse。

 

判别网络最后会判别来自该 mode 的样本是假的。最后，生成网络 G 会简单地锁定到另一个 mode。该循环会无限进行，就会限制生成样本的多样性。

 

#### 2. Convergence（收敛）

 

GAN 训练过程中遇到的一个问题是什幺时候停止训练？因为判别网络 D 损失降级会改善生成网络 G 的损失（反之亦然），因此无法根据损失函数的值来判断收敛，如下图所示：

 

![img](http://flashgene.com/wp-content/uploads/2019/02/0fc4a88506811c0c06286dd1ffe30cd2.jpg)

 

典型的GAN损失函数图。注意该如何从这个图中解释收敛性。

 

#### 3. Quality（质量）

 

与前面提到的收敛问题一样，很难量化地判断生成网络 G 什幺时候会生成高质量的样本。另外，在损失函数中加入感知正则化则在一定程度上可缓解该问题。

 

#### 4. Metrics（度量）

 

GAN 的目标函数解释了生成网络 G 或 判别网络 D 如何根据组件来执行，但它却不表示输出的质量和多样性。因此，需要许多不同的度量指标来进行衡量。

 

### 

 

#### 改善性能的技术

 

下面总结了一些可以使 GAN 更加稳定使用的技术。

 

#### 1. Alternative Loss Functions （替代损失函数）

 

修复 GAN 缺陷的最流行的补丁是  Wasserstein GAN （https://arxiv.org/pdf/1701.07875.pdf）。该 GAN 用 Earth Mover distance ( Wasserstein-1 distance 或 EM distance) 来替换传统 GAN 的 Jensen Shannon divergence ( J-S 散度) 。EM 距离的原始形式很难理解，因此使用了双重形式。这需要判别网络是 1-Lipschitz，通过修改判别网络的权重来维护。

 

使用 Earth Mover distance 的优势在于即使真实的生成数据分布是不相交的，它也是连续的。同时，在生成的图像质量和损失值之间存在一定关系。使用 Earth Mover distance 的劣势在于对于每个生成模型 G  都要执行许多判别网络 D 的更新。而且，研究人员认为权重修改是确保 1-Lipschitz 限制的极端方式。

 

![img](http://flashgene.com/wp-content/uploads/2019/02/5a944279462c91b4f27298fb162d337e.jpg)

 

左图中 earth mover distance 是连续的, 即便其分布并不连续, 这不同于优图中的 the Jensen Shannon divergence。

 

另一个解决方案是使用均方损失（ mean squared loss ）替代对数损失（ log loss ）。LSGAN （https://arxiv.org/abs/1611.04076）的作者认为传统 GAN 损失函数并不会使收集的数据分布接近于真实数据分布。

 

原来 GAN  损失函数中的对数损失并不影响生成数据与决策边界（decision boundary）的距离。另一方面，LSGAN 也会对距离决策边界较远的样本进行惩罚，使生成的数据分布与真实数据分布更加靠近，这是通过将均方损失替换为对数损失来完成的。

 

#### 2. Two Timescale Update Rule (TTUR)

 

在 TTUR 方法中，研究人员对判别网络 D 和生成网络 G 使用不同的学习速度。低速更新规则用于生成网络 G ，判别网络 D使用 高速更新规则。使用 TTUR 方法，研究人员可以让生成网络 G 和判别网络 D 以 1:1 的速度更新。 SAGAN （https://arxiv.org/abs/1805.08318） 就使用了 TTUR 方法。

 

#### 3. Gradient Penalty （梯度惩罚）

 

论文 Improved Training of WGANs（https://arxiv.org/abs/1704.00028）中，作者称权重修改会导致优化问题。权重修改会迫使神经网络学习学习更简单的相似（simpler approximations）达到最优数据分布，导致结果质量不高。同时如果 WGAN 超参数设置不合理，权重修改可能会出现梯度消失或梯度爆炸的问题，论文作者在损失函数中加入了一个简单的梯度惩罚机制以缓解该问题。

 

![img](http://flashgene.com/wp-content/uploads/2019/02/f9f9781cb58ed3ae8777ad5280bf280d.jpg)

 

加入 Gradient Penalty 作为正则化器

 

DRAGAN （https://arxiv.org/abs/1705.07215）的作者称，当 GAN 的博弈达到一个局部平衡态（local equilibrium state），就会出现 mode collapse 的问题。而且判别网络 D 在这种状态下产生的梯度是非常陡（sharp）的。一般来说，使用梯度惩罚机制可以帮助避免这种状态的产生，极大增强 GAN 的稳定性，尽可能减少 mode collapse 问题的产生。

 

#### 4. Spectral Normalization（谱归一化）

 

Spectral normalization 是用在判别网络 D 来增强训练过程的权重正态化技术 （weight normalization technique），可以确保判别网络 D 是 K-Lipschitz 连续的。 SAGAN (https://arxiv.org/abs/1805.08318)这样的实现也在判别网络 D 上使用了谱正则化。而且该方法在计算上要比梯度惩罚方法更加高效。

 

#### 5. Unrolling and Packing (展开和打包)

 

文章 Mode collapse in GANs（http://aiden.nibali.org/blog/2017-01-18-mode-collapse-gans/）中提到一种预防 mode hopping 的方法就是在更新参数时进行预期对抗（anticipate counterplay）。展开的 GAN ( Unrolled GANs ）可以使用生成网络 G 欺骗判别网络 D，然后判别网络 D 就有机会进行响应。

 

另一种预防 mode collapse 的方式就是把多个属于同一类的样本进行打包，然后传递给判别网络 D 。PacGAN （https://arxiv.org/abs/1712.04086）就融入了该方法，并证明可以减少 mode collapse 的发生。

 

#### 6. 多个 GAN

 

一个 GAN 可能不足以有效地处理任务，因此研究人员提出使用多个连续的 GAN ，每个 GAN 解决任务中的一些简单问题。比如，FashionGAN（https://www.cs.toronto.edu/~urtasun/publications/zhu_etal_iccv17.pdf）就使用 2 个 GAN 来执行图像定位翻译。

 

![img](http://flashgene.com/wp-content/uploads/2019/02/4217d591316b3719f80aee8a6dbd40ec.jpg)

 

FashionGAN 使用两个 GANs 进行图像定位翻译。

 

因此，可以让 GAN 慢慢地解决更难的问题。比如 Progressive GANs (ProGANs，https://arxiv.org/abs/1710.10196) 就可以生成分辨率极高的高质量图像。

 

\7. Relativistic GANs（相对生成对抗网络）

 

传统的 GAN 会测量生成数据为真的可能性。Relativistic GANs 则会测量生成数据“逼真”的可能性。研究人员可以使用相对距离测量方法（appropriate distance measure）来测量相对真实性（relative realism），相关论文链接：https://arxiv.org/abs/1807.00734。

 

![img](http://flashgene.com/wp-content/uploads/2019/02/7180d37138f95669d4a6fce67b76f0cb.jpg)

 

图 A 表示 JS 散度的最优解，图 B 表示使用标准 GAN 损失时判别网络 D 的输出，图 C 表示输出曲线的实际图。

 

在论文中，作者提到判别网络 D 达到最优状态时，D 的输出应该聚集到 0.5。但传统的 GAN 训练算法会让判别网络 D 对图像输出“真实”（real，1）的可能性，这会限制判别网络 D 达到最优性能。不过这种方法可以很好地解决这个问题，并得到不错的结果。

 

![img](http://flashgene.com/wp-content/uploads/2019/02/5b2f4ea97d7877bf645b1d60f837cc69.jpg)

 

经过 5000 次迭代后，标准 GAN (左)和相对 GAN (右)的输出。

 

#### 8. Self Attention Mechanism（自注意力机制）

 

Self Attention GANs（https://arxiv.org/abs/1805.08318）作者称用于生成图像的卷积会关注本地传播的信息。也就是说，由于限制性接收域这会错过广泛传播关系。

 

![img](http://flashgene.com/wp-content/uploads/2019/02/336c64231b390ccf81ba2a2d1613d8ba.jpg)

 

将 attention map (在黄色框中计算)添加到标准卷积操作中。

 

Self-Attention Generative Adversarial Network 允许图像生成任务中使用注意力驱动的、长距依赖的模型。自注意力机制是对正常卷积操作的补充，全局信息（长距依赖）会用于生成更高质量的图像，而用来忽略注意力机制的神经网络会考虑注意力机制和正常的卷积。（相关论文链接：https://arxiv.org/pdf/1805.08318.pdf）。

 

![img](http://flashgene.com/wp-content/uploads/2019/02/16dc47488f8c35ef419935208ee79243.jpg)

 

使用红点标记的可视化 attention map。

 

#### 9. 其他技术

 

其他可以用来改善 GAN 训练过程的技术包括：

 

特征匹配

 

Mini Batch Discrimination（小批量判别）

 

历史平均值

 

One-sided Label Smoothing（单侧标签平滑）

 

Virtual Batch Normalization（虚拟批量正态化）











我们知道普通的模型都是搭好架构，然后定义好loss，直接扔给优化器训练就行了。但是GAN不一样，一般来说它涉及有两个不同的loss，这两个loss需要交替优化。现在主流的方案是判别器和生成器都按照1:1的次数交替训练（各训练一次，必要时可以给两者设置不同的学习率，即TTUR），交替优化就意味我们需要传入两次数据（从内存传到显存）、执行两次前向传播和反向传播。

如果我们能把这两步合并起来，作为一步去优化，那么肯定能节省时间的，这也就是GAN的同步训练。

（注：本文不是介绍新的GAN，而是介绍GAN的新写法，这只是一道编程题，不是一道算法题～）

## 如果在TF中[ #](https://kexue.fm/archives/6387#如果在TF中)



如果是在tensorflow中，实现同步训练并不困难，因为我们定义好了判别器和生成器的训练算子了（假设为`D_solver`和`G_solver`），那么直接执行

```python
sess.run([D_solver, G_solver], feed_dict={x_in: x_train, z_in: z_train})
```

就行了。这建立在我们能分别获取判别器和生成器的参数、能直接操作`sess.run`的基础上。

## 更通用的方法[ #](https://kexue.fm/archives/6387#更通用的方法)

但是如果是Keras呢？Keras中已经把流程封装好了，一般来说我们没法去操作得如此精细。所以，下面我们介绍一个通用的技巧，只需要定义单一一个loss，然后扔给优化器，就能够实现GAN的训练。同时，从这个技巧中，我们还可以学习到如何更加灵活地操作loss来控制梯度。

### 判别器的优化[ #](https://kexue.fm/archives/6387#判别器的优化)

我们以GAN的hinge loss为例子，它的形式是：
*D*=argmin*D*E*x*∼*p*(*x*)[max(0,1+*D*(*x*))]+E*z*∼*q*(*z*)[max(0,1−*D*(*G*(*z*)))]*G*=argmin*G*E*z*∼*q*(*z*)[*D*(*G*(*z*))]




注意*G*，因为argmin*D*,*G*。

为了固定*G*，除了“把*G*的参数从优化器中去掉”这个方法之外，我们也可以利用`stop_gradient`去手动固定：
*D*,*G*=argmin*D*,*G*E*x*∼*p*(*x*)[max(0,1+*D*(*x*))]+E*z*∼*q*(*z*)[max(0,1−*D*(*G**n**g*(*z*)))]




这里*G**n**g*(*z*)=stop_gradient(*G*(*z*))
(2)中，我们虽然同时放开了(2)，会变的只有*G*是不会变的，因为我们用的是基于梯度下降的优化器，而*G*的梯度被强行设置为0，所以它的更新量一直都是0。

### 生成器的优化[ #](https://kexue.fm/archives/6387#生成器的优化)

现在解决了*D*的优化，那么*G*呢？`stop_gradient`可以很方便地放我们固定里边部分的梯度（比如*D*(*G*(*z*))的*G*(*z*)），但*G*的优化是要我们去固定外边的*D*，没有函数实现它。但不要灰心，我们可以用一个数学技巧进行转化。

首先，我们要清楚，我们想要*D*(*G*(*z*))里边的*G*的梯度，不想要*D*的梯度，如果直接对*D*(*G*(*z*))求梯度，那么同时会得到*D*,*G*的梯度。如果直接求*D*(*G**n**g*(*z*))的梯度呢？只能得到*D*的梯度，因为*G*已经被停止了。那么，重点来了，将这两个相减，不就得到单纯的*G*的梯度了吗！
*D*,*G*=argmin*D*,*G*E*z*∼*q*(*z*)[*D*(*G*(*z*))−*D*(*G**n**g*(*z*))]




现在优化式*D*是不会变的，改变的是



> 注：不需要从链式法则来理解这种写法，而是要通过stop_gradient本身的意义来理解。对于*L*(*D*,*G*)，不管*G*,*D*的关系是什么，完整的梯度都是(∇*D**L*,∇*G**L*)，而把*G*的梯度停止后，相当于*G*的梯度强行设置为0的，也就是*L*(*D*,*G**n**g*)的梯度实际上为(∇*D**L*,0)，所以*L*(*D*,*G*)−*L*(*D*,*G**n**g*)的梯度是(∇*D**L*,∇*G**L*)−(∇*D**L*,0)=(0,∇*G**L*)。

值得一提的是，直接输出这个式子，结果是恒等于0，因为两部分都是一样的，直接相减自然是0，但它的梯度不是0。也就是说，这是一个恒等于0的loss，但是梯度却不恒等于0。

### 合成单一loss[ #](https://kexue.fm/archives/6387#合成单一loss)

好了，现在式(2)和式(4)都同时放开了*D*,*G*，大家都是argmin，所以可以将两步合成一个loss：
*D*,*G*=argmin*D*,*G*E*x*∼*p*(*x*)[max(0,1+*D*(*x*))]+E*z*∼*q*(*z*)[max(0,1−*D*(*G**n**g*(*z*)))]+*λ*E*z*∼*q*(*z*)[*D*(*G*(*z*))−*D*(*G**n**g*(*z*))]




写出这个loss，就可以同时完成判别器和生成器的优化了，而不需要交替训练，但是效果基本上等效于1:1的交替训练。引入1:*λ*。

**参考代码：**https://github.com/bojone/gan/blob/master/gan_one_step_with_hinge_loss.py

## 文章小结[ #](https://kexue.fm/archives/6387#文章小结)

文章主要介绍了实现GAN的一个小技巧，允许我们只写单个模型、用单个loss就实现GAN的训练。它本质上就是用`stop_gradient`来手动控制梯度的技巧，在其他任务上也可能用得到它。

所以，以后我写GAN都用这种写法了，省力省时～当然，理论上这种写法需要多耗些显存，这也算是牺牲空间换时间吧。


  