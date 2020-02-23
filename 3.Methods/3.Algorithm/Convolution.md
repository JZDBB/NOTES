# Deformable Convolution

## separable convolutions

- 空间可分离卷积（spatial separable convolutions）：空间可分离就是将$3 \times 3$的卷积核拆分为$3 \times 1$和$1\times 3$的卷积核的两步操作。
- 深度可分离卷积（depthwise separable convolutions）： 深度可分离卷积的过程可以分为两部分：深度卷积（depthwise convolution）和逐点卷积（pointwise convolution）。 

#### 深度可分离卷积

**深度卷积** 在不改变深度的情况下对输入图像进行卷积。我们使用3个形状为$5 \times 5 \times 1$的内核。  每个$5 \times 5 \times 1$内核迭代图像的一个通道，得到每$25$个像素组的标量积，得到一个$8 \times 8 \times 1$图像。将这些图像叠加在一起可以创建一个$8 \times 8 \times 3$的图像。

<img src="../img/separate conv1.png" height="200px">

  **逐点卷积** 增加每个图像的通道数  通过$8 \times 8 \times 3$图像迭代$1 \times 1 \times 3$内核，得到$8 \times 8 \times 1$图像。  

<img src="../img/separate conv2.png" height="200px">

创建$256$个$1 \times 1 \times 3$内核，每个内核输出一个$8 \times 8 \times 1$图像，以得到形状为$8 \times 8 \times 256$的最终图像。 

<img src="../img/separate conv3.png" height="200px">

也就是 把卷积分解成两部分：深度卷积和逐点卷积。更抽象地说，如果原始卷积函数是$12 \times 12 \times 3 - (5 \times 5 \times 3 \times 256)→8 \times 8 \times 256$，我们可以将这个新的卷积表示为$12 \times 12 \times 3 - (5 \times 5 \times 1 \times 1) → (1 \times 1 \times 3 \times 256) →8 \times 8 \times 256$。  

**深度可分离卷积的意义** 

普通卷积有$256$个$5 \times 5 \times 3$内核可以移动$8 \times 8$次。这是$256 \times 3 \times 5 \times 5 \times 8 \times 8 = 1228800$。 可分离卷积呢?在深度卷积中，我们有$3$个$5 \times 5 \times 1$的核它们移动了$8 \times 8$次。也就是$3 \times 5 \times 5 \times 8 \times 8 = 4800$。在点态卷积中，我们有$256$个$1 \times 1 \times 3$的核它们移动了$8 \times 8$次。这是$256 \times 1 \times 1 \times 3 \times 8 \times 8 = 49152$乘法。加起来是$53952次$乘法。 计算量越少，网络就能在更短的时间内处理更多的数据。 

速度上的区别主要是：在普通卷积中，进行了$256$次变换。每个变换都要用到$5 \times 5 \times 3 \times 8 \times 8=4800$次乘法。在可分离卷积中，只对图像做一次变换——在**深度卷积**中。然后，我们将转换后的图像简单地延长到$256$通道。在Tensorflow中，都有一个称为“**深度乘法器**”的参数默认设置为1。改变这个参数，可以改变深度卷积中输出通道的数量。如，将深度乘法器设置为$2$，每个$5 \times 5 \times 1$内核将输出$8 \times 8 \times 2$的图像，使深度卷积的总输出(堆叠)为$8 \times 8 \times 6$，而不是$8 \times 8 \times 3$。选择手动设置深度乘法器来增加神经网络中的参数数量，更好地学习更多的特征。 

**缺点**在于：它减少了卷积中参数的数量，如果网络已经很小，参数减少导致欠拟合可能会无法训练出好的结果。如果使用得当，它可以在不显著降低效力的情况下提高效率。 

**$1 \times 1$内核:** 

一个$1 \times 1$内核——或者更确切地说，$n$个$1 \times 1 \times m$内核，其中$n$是输出通道的数量，$m$是输入通道的数量——可以在可分离卷积之外使用。$1 \times 1$内核目的是增加或减少图像的深度。同时，$1 \times 1$核的主要目的是应用非线性。在神经网络的每一层之后，我们都可以应用一个激活层。无论是$ReLU、PReLU、Softmax$，与卷积层不同，激活层是非线性的。非线性层扩展了模型的可能性，这也是通常使“深度”网络优于“宽”网络的原因。为了在不显著增加参数和计算量的情况下增加非线性层的数量，我们可以应用一个1x1内核并在它之后添加一个激活层。这有助于给网络增加一层深度。 







##  （Group convolution）、空洞卷积（Dilated convolution 或 À trous） 