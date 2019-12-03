# **训练 GAN 的常用策略**

## Training Tips

**1. model choice**

- 如果你不知道选择什么样的模型，那就选择 DCGAN$^{[3] }$或者 ResNet$^{[4]}$ 作为 base model。能采用DCGAN尽量用，不行尽量采用Hybri模型：KL+GAN、VAE+GAN	
- 采用更大的卷积核以及更多卷积：1. less filters $ \longrightarrow$ more blurry. more filters help capture additional information which can eventually add sharpness to the generated images. 2. larger kernels$\longrightarrow$ bigger receptive field. larger kernels at the top convolutional layers to maintain some kind of smoothness.

**2. input layer**

- 输入是一张图片，将图片数值归一化到$ [-1, 1]$
- 输入是一个随机噪声的向量，最好是从$$N(0, 1)$$ 的正态分布里面采样，不要从$$U(0,1)$$ 的均匀分布里采样。

**3. output layer**

使用输出通道为 3 的卷积作为最后一层，可以采用 $1\times1 $或者 $3\times3$ 的 filters，有的论文也使用 $9\times9$ 的
filters。（注：ganhacks 推荐使用 tanh）

Ps$^{[25]}$:

- 将输入规范化到 -1 和 1 之间
- G 的输出层采用`Tanh`激活函数

**4. Hidden layer**

- 稀疏梯度会影响 GAN 的稳定性
- 在 G 和 D 中采用 LeakyReLU 代替 Relu 激活函数
- 对于下采样操作，可以采用平均池化(Average Pooling) 和 Conv2d+stride 的替代方案——Avoid max pooling for downsampling. Use convolution stride.
- 对于上采样操作，可以使用 [PixelShuffle](https://arxiv.org/abs/1609.05158), ConvTranspose2d + stride在做 decode 的时候，尽量使用 upsample+conv2d 组合代替 transposed_conv2d，可以减少 checkerboard的产生$^{[5]；}$在做超分辨率等任务上，可以采用 pixelshuffle$^{[6]。}$在 tensorflow 里，可以用 tf.depth_to_sapce 来实现pixelshuffle 操作。Use PixelShuffle and transpose convolution for upsampling.

**5. normalization**

虽然在 resnet 里的标配是 BN，在分类任务上表现很好，但是图像生成方面，推荐使用其他 normlization 方法，例如parameterized 方法有 instance normalization$^{[8]、}$layer normalization$^{[9] }$等，non-parameterized 方法推荐使用 pixel normalization$^{[10]}$。假如你有选择困难症，那就选择大杂烩的 normalization
方法——switchable normalization$^{[11]}$。

- 采用 mini-batch BatchNorm，要保证每个 mini-batch 都是同样的真实图片或者是生成图片
- 不采用 BatchNorm 的时候，可以采用 instance normalization（对每个样本的规范化操作）
- 可以使用**虚拟批量归一化**(virtural batch normalization):开始训练之前预定义一个 batch R，对每一个新的 batch X，都使用 R+X 的级联来计算归一化参数

**6. discriminator**

想要生成更高清的图像，推荐 multi-stage discriminator$^{[10]}$。简单的做法就是对于输入图片，把它下采样（maxpooling）到不同 scale 的大小，输入三个不同参数但结构相同的discriminator。

**7. minibatch discriminator**

由于判别器是单独处理每张图片，没有一个机制能告诉 discriminator 每张图片之间要尽可能的不相似，这样就会导致判别器会将所有图片都 push到一个看起来真实的点，缺乏多样性。minibatch discriminator [22] 就是这样这个机制，显式地告诉 discriminator每张图片应该要不相似。在 tensorflow 中，一种实现 minibatch discriminator 方式如下：

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

上面是通过一个可学习的网络来显示度量每个样本之间的相似度，PGGAN里提出了一个更廉价的不需要学习的版本，即通过统计每个样本特征每个像素点的标准差，然后取他们的平均，把这个平均值复制到与当前 feature map
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

**8. loss**

- **Generator loss的误差曲线走向**
  因为Generator的loss定义为： G_loss = -tf.reduce_mean(D_fake) Generator_loss = gen_loss + lamda*G_loss：其中gen_loss为Generator的loss，G_loss为Discrimnator的loss，目标是使Generator_loss不断变小。所以理想的Generator loss的误差曲线应该是不断往0靠的下降的抛物线。
- **Discrimnator loss的误差曲线走向**
  因为Discrimnator的loss定义为： D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)这个是一个和Generator抗衡的loss。目标就是使判别器分不清哪个是生成器的输出哪个是真实的label。所以理想的Discrimnator loss的误差曲线应该是最终在0附近振荡，即傻傻分不清。换言之，就是判别器有50%的概率判断你是真的，50%概率判断你是假的。

通常情况下，GAN loss 配合以下几种 loss，效果会更好。

- perceptual loss [17]
- style loss [18]
- total variation loss [17]
- l1 reconstruction loss

**9. gradient penalty**

Gradient penalty 首次在 wgan-gp 里面提出来的，记为 1-gp，目的是为了让 discriminator 满足1-lipchitchz 连续，后续 Mescheder, Lars M. et al$^{[19]} $又提出了只针对正样本或者负样本进行梯度惩罚，记为 0-gp-sample。Thanh-Tung, Hoang et al$^{[20]}$ 提出了 0-gp，具有更好的训练稳定性。

code for Tensorflow

```python
# tensorflow-version
    # Standard WGAN loss
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    # Gradient penalty
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty
```

code for Pytorch

```python
# pytorch-version
gradient_penalty = _gradient_penalty(data, generated_data)
def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,                           grad_outputs=torch.ones(prob_interpolated.size()).cuda() if 						self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
```

**10. Spectral normalization$^{[21]}$**

谱归一化是另外一个让判别器满足 1-lipchitchz 连续的利器，建议在判别器和生成器里同时使用。Spectral Normalization 是一种**权重归一化技术**，通常用于鉴别器上，以增强训练过程。这本质上保证了鉴别器是 **K-Lipschitz** 连续的。像 SAGAN 这样的一些实现，也在生成器上使用 spectral Normalization。该方法比梯度惩罚法计算效率更高。

**11. one-size label smoothing$^{[22]}$**

平滑正样本的 label，例如 label 1 变成 0.8-1.2 之间的随机数，保持负样本 label 仍然为 0。但是在训练G的时候不做平滑。个人经验表明这个 trick能够有效缓解训练不稳定的现象，但是不能根本解决问题，假如模型不够好的话，随着训练的进行，后期 loss 会飞。

**12. add supervised labels**

- add labels
- conditional batch normalization

**13. instance noise (decay over time)**

在原始 GAN 中，我们其实在优化两个分布的 JS 散度，前面的推理表明在两个分布的支撑集没有交集或者支撑集是低维的流形空间，他们之间的 JS散度大概率上是 0；而加入 instance noise 就是强行让两个分布的支撑集之间产生交集，这样 JS 散度就不会为 0。新的 JS 散度变为：
$$
d_{\sigma,JS(P_r|P_g)=JS[P_{\sigma}\times P_r|P_{\sigma}\times P_g]}
$$
**14. TTUR$^{[23]}$**

在优化 G 的时候，我们默认是假定我们的 D 的判别能力是比当前的 G 的生成能力要好的，这样 D 才能指导 G 朝更好的方向学习。通常的做法是先更新 D的参数一次或者多次，然后再更新 G 的参数，TTUR 提出了一个更简单的更新策略，即分别为 D 和 G 设置不同的学习率，让 D 收敛速度更快。通常，生成器使用较慢的更新规则 (update rule)，鉴别器使用较快的更新规则。使用这种方法，我们可以以 1:1 的比例执行生成器和识别器的更新，只需要修改学习率。

**15. training strategy**

- PGGAN$^{[10]}$

  PGGAN 是一个渐进式的训练技巧，因为要生成高清（$1024 \times 1024$）的图片，直接从一个随机噪声生成这么高维度的数据是比较难的；既然没法一蹴而就，那就循序渐进，首先从简单的低纬度的开始生成，例如
  $4 \times 4$，然后 $16 \times 16$，直至我们所需要的图片大小。此外，由于大部分的操作都是在比较低的维度上进行的，训练速度也不比其他模型逊色多少。

- coarse-to-refine

  coarse-to-refine 可以说是 PGGAN 的一个特例，它的做法就是先用一个简单的模型，加上一个 l1-loss，训练一个模糊的效果，然后再把这个模糊的照片送到后面的 refine 模型里，辅助对抗 loss 等其他loss，训练一个更加清晰的效果。这个在图片生成里面广泛应用。

**16. Exponential Moving Average$^{[24]}$**

EMA 主要是对历史的参数进行一个指数平滑，可以有效减少训练的抖动。强烈推荐！！！

**17. Optimizer**

- 使用Adam，Adam的优化效率对于GAN很显著，前提是能用则用，像WGAN那样，规定不能使用，才考虑替换。
- 也可以使用SGD作为鉴别器优化，Adam作为生成器优化。

##### 18. 尽早追踪失败的原因

- D 的 loss 变成 0，那么这就是训练失败了
- 检查规范的梯度：如果超过 100，那出问题了
- 如果训练正常，那么 D loss 有低方差并且随着时间降低
- 如果 g loss 稳定下降，那么它是用糟糕的生成样本欺骗了 D

##### 19. Feature Matching$^{[26]}$

Feature matching changes the cost function for the generator to minimizing the statistical difference between the features of the real images and the generated images. Often, we measure the L2-distance between the means of their feature vectors. Therefore, feature matching expands the goal from beating the opponent to matching features in real images. Here is the new objective function:
$$
\left\|\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }} \mathbf{f}(\boldsymbol{x})-\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}}(\boldsymbol{z})} \mathbf{f}(G(\boldsymbol{z}))\right\|_{2}^{2}
$$
where *f(x)* is the feature vector extracted in an immediate layer by the discriminator.

<img src=".\img\feature matching.jpeg" height=250px>

The means of the real image features are computed per minibatch which fluctuate on every batch. It is good news in mitigating the mode collapse. It introduces randomness that makes the discriminator harder to overfit itself.

> Feature matching is effective when the GAN model is unstable during training.





## Reference

[1]. Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.

[2]. Arjovsky, Martín and Léon Bottou. “Towards Principled Methods for Training Generative Adversarial Networks.” CoRR abs/1701.04862 (2017): n. pag.

[3]. Radford, Alec et al. “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.” CoRR abs/1511.06434 (2016): n. pag.

[4]. He, Kaiming et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 770-778.

[5]. https://distill.pub/2016/deconv-checkerboard/

[6]. Shi, Wenzhe et al. “Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 1874-1883.

[7]. Yu, Jiahui et al. “Free-Form Image Inpainting with Gated Convolution.” CoRRabs/1806.03589 (2018): n. pag.

[8]. Ulyanov, Dmitry et al. “Instance Normalization: The Missing Ingredient for Fast Stylization.” CoRR abs/1607.08022 (2016): n. pag.

[9]. Ba, Jimmy et al. “Layer Normalization.” CoRR abs/1607.06450 (2016): n. pag.

[10]. Karras, Tero et al. “Progressive Growing of GANs for Improved Quality, Stability, and Variation.” CoRR abs/1710.10196 (2018): n. pag.

[11]. Luo, Ping et al. “Differentiable Learning-to-Normalize via Switchable Normalization.” CoRRabs/1806.10779 (2018): n. pag.

[12]. Arjovsky, Martín et al. “Wasserstein GAN.” CoRR abs/1701.07875 (2017): n. pag.

[13]. Mao, Xudong, et al. "Least squares generative adversarial networks." Proceedings of the IEEE International Conference on Computer Vision. 2017.

[14]. Zhang, Han, et al. "Self-attention generative adversarial networks." arXiv preprint arXiv:1805.08318 (2018).

[15]. Brock, Andrew, Jeff Donahue, and Karen Simonyan. "Large scale gan training for high fidelity natural image synthesis." arXiv preprint arXiv:1809.11096 (2018).

[16]. Gulrajani, Ishaan et al. “Improved Training of Wasserstein GANs.” NIPS (2017).

[17]. Johnson, Justin et al. “Perceptual Losses for Real-Time Style Transfer and Super-Resolution.” ECCV (2016).

[18]. Liu, Guilin et al. “Image Inpainting for Irregular Holes Using Partial Convolutions.” ECCV(2018).

[19]. Mescheder, Lars M. et al. “Which Training Methods for GANs do actually Converge?” ICML(2018).

[20]. Thanh-Tung, Hoang et al. “Improving Generalization and Stability of Generative Adversarial Networks.” CoRR abs/1902.03984 (2018): n. pag.

[21]. Yoshida, Yuichi and Takeru Miyato. “Spectral Norm Regularization for Improving the Generalizability of Deep Learning.” CoRR abs/1705.10941 (2017): n. pag.

[22]. Salimans, Tim et al. “Improved Techniques for Training GANs.” NIPS (2016).

[23]. Heusel, Martin et al. “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.” NIPS (2017).

[24]. Yazici, Yasin et al. “The Unusual Effectiveness of Averaging in GAN Training.” CoRRabs/1806.04498 (2018): n. pag.

[25]. [Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)

[26]. [GAN — Ways to improve GAN performance](https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b)











**2、收敛 (Convergence)**



GAN 训练中一个常见的问题是 “**我们应该在什么时候停止训练？**”。由于鉴别器损失降低时，生成器损失增加 (反之亦然)，我们不能根据损失函数的值来判断收敛性。如下图所示:



![img](http://cdn.zhuanzhi.ai/images/wx/7e3a7e59cdc8dcf7fb82e4c810f8234c)

一个典型的 GAN 损失函数







**3、梯度惩罚 (Gradient Penalty)**



在 Improved Training of WGANs 这篇论文中，作者声称 **weight clipping** 会导致优化问题。



作者表示， weight clipping 迫使神经网络学习最优数据分布的 “更简单的近似”，从而导致较低质量的结果。他们还声称，如果没有正确设置 WGAN 超参数，那么 weight clipping 会导致梯度爆炸或梯度消失问题。



作者在损失函数中引入了一个简单的 **gradient penalty**，从而缓解了上述问题。此外，与最初的 WGAN 实现一样，保留了 1-Lipschitz 连续性。



![img](http://cdn.zhuanzhi.ai/images/wx/ba6bebf0333e07641036da9dec1bb992)

与 WGAN-GP 原始论文一样，添加了 gradient penalty 作为一个正则化器



DRAGAN 的作者声称，当 GAN 所玩的游戏达到 “局部平衡状态” 时，就会发生 mode collapse。他们还声称，鉴别器围绕这些状态产生的梯度是“尖锐的”。当然，使用 gradient penalty 可以帮助我们避开这些状态，大大增强稳定性，减少模式崩溃。

**5、Unrolling 和 Packing**

防止 mode hopping 的一种方法是预测未来，并在更新参数时预测对手。Unrolled GAN 使生成器能够在鉴别器有机会响应之后欺骗鉴别器。

防止 mode collapse 的另一种方法是在将属于同一类的多个样本传递给鉴别器之前 “打包” 它们，即 **packing**。这种方法被 PacGAN 采用，在 PacGAN 论文中，作者报告了 mode collapse 有适当减少。

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



训练WGAN的时候，有几个方面可以调参：

   a. 调节Generator loss中GAN loss的权重。 G loss和Gan loss在一个尺度上或者G loss比Gan loss大一个尺度。但是千万不能让Gan loss占主导地位, 这样整个网络权重会被带偏。

 

### 启发式技巧是启发式平均（heuristic averaging），如果网络参数偏离之前值的运行平均值，则会受到惩罚，这有助于收敛到平衡态。







### **Improved Techniques for training GANs：**

Feature matching：利用中间层feature map增加了一个新的损失函数，加速了平衡的收敛

minibatch discrimination：解决mode collapse问题

historical averaging：使参数更新时候不会忘了之前由其他样本得出的经验

one-sided label smoothing：reduce the vulnerability of neural networks to adversarial examples

virtual batch normalization：batch normalization的缺陷是造成神经网络对于输入样本的输出值极大依赖于在同一个batch中的其他样本，该技巧选择了固定的batch生成statistics来normalize输入样本

 	


