## mode collapse & 泛化性

“泛化性” 它能不能生成和发现**新**的样本和变化模式，而不仅仅是**记住**已有的样本和模式。泛化能力对提高基于GAN的半监督模型的性能也是非常重要的，因为只要具有泛化能力的模型，才能产生有价值的新样本来增强训练集，并帮助模型通过挖掘出更有信息的类内和类间变化来提高小样本情况下的分类准确度。

 通过训练得到的生成器G(z)，其实就是一个非常好的流型模型。这里z就是流型上的参数坐标，通过不断变化z，我们就可以在高维空间中划出一个流型结构。   **从应用的角度，给定了一个图像x，用局部表示G(x,z)可以对这个x在它的局部领域中做各种编辑操作或者控制图像的各种属性**。这个可以结合有监督的对局部参数的意义进行训练。 



解释和避免**GAN的mode collapse**。具体来说，给定了一个z，当z发生变化的时候，对应的G(z)没有变化，那么在这个局部，GAN就发生了mode collapse，也就是不能产生不断连续变化的样本。  最直接的是我们可以给流型的切向量加上一个正交约束(Orthonormal constraint)，从而避免这种局部的维度缺陷。 对mode collapse这个典型想象的理解，可以有助于为研究生成器的泛化性提供非常有价值的样本。比如，在发生mode collapse的地方，该点的密度函数显然会存在一个很高的峰值，在这个峰附近，密度函数的Lipschitz常数会变得很大。这个提示我们，通过对生成密度做Lipschitz正则化，是有利于帮助我们解决mode collapse问题，同时提高生成器的泛化性的。LS-GAN的成功也证实了这点。 而另一方面，我们上面介绍的通过对切向量做正交约束，进而防止流型维度缺陷和mode collapse的方法， 是不是也能为我们打开一扇**从几何角度**提高生成器泛化性的思路呢？

### 解决思路

1. 提升 GAN 的学习能力，进入更好的局部最优解，如下图所示，通过训练红线慢慢向蓝线的形状、大小靠拢，比较好的局部最优自然会有更多的模式，直觉上可以一定程度减轻模式崩溃的问题。 

2. 放弃寻找更优的解，只在 GAN 的基础上，显式地要求 GAN 捕捉更多的模式（如下图所示），虽然红线与蓝线的相似度并不高，但是“强制”增添了生成样本的多样性，而这类方法大都直接修改 GAN 的结构。

<img src="./img/mode collapse1.jpg" height=250px>

#### MAD-GAN

即使单个生成器会产生模式崩溃的问题，但是如果同时构造多个生成器，且让每个生成器产生不同的模式，则这样的多生成器结合起来也可以保证产生的样本具有多样性，多个生成器彼此“联系”，不同的生成器尽量产生不相似的样本，而且都能欺骗判别器。

在 MAD（Multi-agent diverse）GAN 中，共包括 k 个初始值不同的生成器和 1 个判别器，与标准 GAN 的生成器一样，每个生成器的目的仍然是产生虚假样本试图欺骗判别器。对于判别器，它不仅需要分辨样本来自于训练数据集还是其中的某个生成器，而且还需要驱使各个生成器尽量产生不相似的样本。

需要将判别器做一些修改：将判别器最后一层改为 k+1 维的 softmax 函数，对于任意输入样本 x，D(x) 为 k+1 维向量，其中前 k 维依次表示样本 x 来自前 k 个生成器的概率，第 k+1 维表示样本 x 来自训练数据集的概率。D 的目标函数应为最小化 D(x) 与 delta 函数的交叉熵：
$$
\max \limits_{\theta_{d}}\mathbb{E}_{x\sim p}-H(\delta,D(x))
$$
直观上看，这样的损失函数会迫使每个 x 尽量只产生于其中的某一个生成器，而不从其他的生成器中产生，将其展开则为：
$$
\max \limits_{\theta_{d}}\mathbb{E}_{x\sim p_{data}}log[D_{k+1}(x)]+\sum^k_{i=1}\mathbb{E}_{x_i \sim p_{g_i}}log[D_i{x_i}]
$$
生成器目标函数为：
$$
\min \limits_{\theta_{g_i}}\mathbb{E}_{z\sim p_z}log[1-D_{k+1}(G_i(z))]
$$
对于固定的生成器，最优判别器为：
$$
D_{k+1}(x)=\frac{p_{data}(x)}{p_{data}(x)+\sum^k_{i=1}p_{g_i}(x)} \qquad D_i(x)=\frac{p_{g_i}(x)}{p_{data}(x)+\sum^k_{i=1}p_{g_i}(x)}
$$
其形式几乎同标准形式的 GAN 相同，只是不同生成器之间彼此“排斥”产生不同的样本。另外，可以证明当
$$
p_{data}(x)=\frac{1}{k}\sum^k_{i=1}p_{g_i}
$$
达到最优解，再一次可以看出，MAD-GAN 中并不需要每个生成器的生成样本概率密度函数逼近训练集的概率密度函数，每个生成器都分别负责生成不同的样本，只须保证生成器的平均概率密度函数等于训练集的概率密度函数即可。