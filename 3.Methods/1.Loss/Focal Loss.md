# Focal Loss

[Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf) 

本质上讲，Focal Loss 就是一个解决**分类问题中类别不平衡、分类难度差异**的一个 loss，其最早提出是由于在目标检测中two-stage detector和one-stage detector的优劣中，为了在one-stage detector中实现和two-stage detector中的准确率。

## 截断loss

分类问题中的交叉熵：
$$
H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = -\sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)},
$$
假设训练数据集的样本数为$n​$，交叉熵损失函数定义为:
$$
\ell(\boldsymbol{\Theta}) = \frac{1}{n} \sum_{i=1}^n H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ),
$$
在二分类中可以表示为：
$$
Loss=-\boldsymbol y\cdot log(\boldsymbol {\hat y})-(1-\boldsymbol y)\cdot log(1-\boldsymbol{\hat y})=\begin{cases}
-log(\boldsymbol {\hat y}),& y=1 \\
-log(1-\boldsymbol {\hat y}),& y=0
\end{cases}
$$
其中$y\in{0, 1}$  是真实标签，${\hat y}$是预测值。

#### 硬截断$Loss$

$$
L^*=\lambda(\boldsymbol y, \boldsymbol {\hat y}) \cdot Loss \quad \Rightarrow \quad
\lambda(\boldsymbol y, \boldsymbol {\hat y})=\begin{cases}
0,& \boldsymbol y=1\ and \ \boldsymbol{\hat y}>0.5 \ or \ \boldsymbol y=0\ and \ \boldsymbol{\hat y}<0.5 \\
1,& otherwise
\end{cases}
$$

这样的做法就是：**正样本的预测值大于 0.5 的，或者负样本的预测值小于 0.5 的，都不更新，调整阈值，把注意力集中在预测不准的那些样本。**这样做能部分地达到目的，但是所需要的迭代次数会大大增加。这是因为，以正样本为例，**我只告诉模型正样本的预测值大于 0.5 就不更新了，却没有告诉它要“保持”大于 0.5**，所以下一阶段，它的预测值就很有可能变回小于 0.5 了，下一回合它又被更新了，这样反复迭代，理论上也能达到目的，但是迭代次数会大大增加。

所以，重点就是**“不只是要告诉模型正样本的预测值大于0.5就不更新了，而是要告诉模型当其大于0.5后就只需要保持就好了”**。

#### 软化$Loss$

硬截断的缺点是$\lambda(\bold y, \bold {\hat y})$是不可导的，软化就是讲一些不可导的函数用可导的函数来表示近似，也可以称为**“光滑化”**，因此改写$Loss$
$$
L^*=\begin{cases}
-\theta(0.5-\boldsymbol{\hat y})\cdot log(\boldsymbol {\hat y}),& \boldsymbol y=1\\
-\theta(\boldsymbol{\hat y}-0.5)\cdot log(1-\boldsymbol {\hat y}),& \boldsymbol y=0
\end{cases}
$$
此处的$\theta​$是一个单位阶跃函数。做软化后可以用**sigmoid函数**替代。
$$
L^*{^*}=\begin{cases}
-\sigma(-\boldsymbol{K}x)\cdot log(\sigma(x)),& \boldsymbol y=1\\
-\sigma(\boldsymbol{K}x)\cdot log(\sigma(-x)),& \boldsymbol y=0
\end{cases}
$$

## Focal Loss

Focal Loss改进的表达式如下：
$$
FL(p_t)=−{(1−p_t)}^γlog(p_t)
$$
其中，
$$
p_t=\begin{cases}
p, &y=1\\
1-p, & otherwise
\end{cases}
$$
这里的$focusing \ parameter \  \gamma >0$，由此才能更加注重那些困难的，错分的数据。

举个例子，设置$\gamma=2​$，对于正样本而言，预测结果为0.95的简单样本，$(1-0.95)^{\gamma}​$是很小的，此时的损失函数就会较小，关注度就减少，更新权值的时候梯度就很小。如果预测结果是0.3，那么$(1-0.3)^{\gamma}​$就会很大。对于负样本来说，也是一样的。因此$(1-p_t)^{\gamma}​$被称之为$modulating\ factor​$（调制系数）。

为了平衡正负样本不均匀的情况，加入了$\alpha$作为平衡因子：
$$
FL(p_t)=\begin{cases}
−{\alpha}{(1−\hat y)}^γlog(\hat y), & y=1\\
−(1-{\alpha}){\hat y}^γlog(1-\hat y), & y=0
\end{cases}
$$
$\gamma$调节简单样本权重降低的速率，当$\gamma$为0时即为交叉熵损失函数，当$\gamma$增加时，调整因子的影响也在增加。实验发现$\gamma$为2是最优。

作者设计focal loss的主要目的是解决one-stage目标检测中前景和背景类别不平衡的问题。同时设计了一个简单密集型网络RetinaNet来训练在保证速度的同时达到了精度最优。

## 多分类Loss

focal loss在多分类中的形式也很容易得到：
$$
FL=-(1-{\hat y_t})^{\gamma}log(\hat y_t)
$$
其中，${\hat y_t}$是目标的预测值，一般就是经过softmax后的结果。

软化后的Loss：
$$
L^*{^*}=-softmax(-K{x_t})^{\gamma}log(softmax(x_t))
$$
这里$x_t$也是目标的预测值，但它是softmax前的结果。