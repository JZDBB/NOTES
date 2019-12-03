## Inception score（IS）

**PS：** **高inception score从逻辑上也只是个高质量图片的必要非充分条件。** 

IS uses two criteria in measuring the performance of GAN:

- The quality of the generated images（质量）
- their diversity.（多样性）

1. 清晰度：把生成的图片$ \bold x$输入 Inception V3 中，将输出1000维的向量 $y$，向量的每个维度的值对应图片属于某类的概率。**对于一个清晰的图片，它属于某一类的概率应该非常大，而属于其它类的概率应该很小**（*这个假设本身是有问题的，有可能有些图片很清晰，但是具体属于哪个类却是模棱两可的*）。用专业术语说，$p(y|\bold x)$  的熵应该很小（熵代表混乱度，均匀分布的混乱度最大，熵最大）。
2. 多样性：如果一个模型能生成足够多样的图片，那么它生成的图片在各个类别中的分布应该是平均的，假设生成了 10000 张图片，那么最理想的情况是，1000类中每类生成了10张。转换成术语，就是生成图片在所有类别概率的边缘分布 $p(y)$熵很大（均匀分布）。具体计算时，可以先用生成器生成 N 张图片，然后用下面公式的经验分布来代替：

$$
\hat p(y)=\frac{1}{N}\sum_{i=1}^{N}p(y|{\bold x}^{(i)})
$$

综合两方面因素，IS的公式可表示为：
$$
\bold {IS}(G)=\bold {exp}(E_{\bold x \sim p_g}D_{KL}(p(y|{\bold x}^{(i)})||p(y)))
$$
$\bold x \sim p_g$：从生成器中生图片。

$p(y|\bold x)$：表示生成图片输入到网络中得到的1000维向量的概率分布，IS 提出者的假设是，对于清晰的生成图片，这个向量的某个维度值格外大，而其余的维度值格外小（也就是概率密度图十分尖）。

$p(y)$：N个生成的图片（N 通常取5000），每个生成图片都输入到 Inception V3 中，各自得到一个自己的概率分布向量，把这些向量求一个平均，得到生成器生成的图片全体在所有类别上的边缘分布。

$D_{KL}$： KL散度。KL 散度用以衡量两个概率分布的距离，它是非负的，值越大说明这两个概率分布越不像。$D_{KL}(P|Q)=\sum_{i}P(i)log\frac{P(i)}{Q(i)}$

综合起来，只要$p(y|\bold x)$和$p(y)$的距离足够大，就证明生产模型足够好。应为前者是个尖锐分布，后者为均匀分布。

#### IS推导

$$
\begin{align}
log(\bold {IS}(G))=E_{\bold x \sim p_g}D_{KL}(p(y|{\bold x}^{(i)})||p(y))\\
=\sum_{\bold x}p_g(\bold x)\sum_ip(y=i|\bold x)log\frac{p(y=i|\bold x)}{p(y=i)}\\
= \sum_{\bold x, i}p(\bold x, y=i)log\frac{p(y=i|\bold x)p(\bold x)}{p(y=i)p(\bold x)}\\
= \sum_{\bold x, i}p(\bold x, y=i)log\frac{p(\bold x, y=i)}{p(y=i)p(\bold x)}
\end{align}
$$



## FID



## MRE



## MS-SSIM

 关于度量泛化性的指标,我在[Conditional Image Synthesis with Auxiliary Classifier GANs](http://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1610.09585.pdf) 这篇文章里看到他们提出用multi-scale structural similarity (MS-SSIM)来衡量GAN输出结果的多样性, 不知道您是否有了解过这方面的工作. 