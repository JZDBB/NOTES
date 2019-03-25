# GAN

## LSGAN

Least Squares Generative Adversarial Networks (LSGANs) 这种方法将辨别器采用最小乘方的损失函数方法，因为这种方法可以惩罚离正确的决策边缘远的地方的样本，同时，最小化目标函数比最小化Pearson χ 2 divergence更好。相当于改进了一个损失函数，后来改进还有log的方法。

最小二乘GAN，正如它的名字所指示的，目标函数将是一个平方误差，考虑到D网络的目标是分辨两类，如果给生成样本和真实样本分别编码为![a,b](https://www.zhihu.com/equation?tex=a%2Cb)，那么采用平方误差作为目标函数，D的目标就是

$\mathop{min}_D L(D)=E_{x \sim p_x}(D(x)-b)^2+E_{z \sim p_z}(D(G(z)-a)^2$

G的目标函数将编码![a](https://www.zhihu.com/equation?tex=a)换成编码![c](https://www.zhihu.com/equation?tex=c)，这个编码表示D将G生成的样本当成真实样本，

$\mathop{min}_G L(G)=E_{z \sim p_z}(D(G(z)-c)^2​$

在下一节我们会证明，当![b-c=1, b-a=2](https://www.zhihu.com/equation?tex=b-c%3D1%2C+b-a%3D2)时，目标函数等价于皮尔森卡方散度（Pearson ![\chi^2](https://www.zhihu.com/equation?tex=%5Cchi%5E2) divergence）。一般来说，取![a=-1, b=1, c=0](https://www.zhihu.com/equation?tex=a%3D-1%2C+b%3D1%2C+c%3D0)或者![a=-1, b=c=1](https://www.zhihu.com/equation?tex=a%3D-1%2C+b%3Dc%3D1)。

## LS-GAN