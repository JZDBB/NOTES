# Light Weight CNN

 构建light weight CNN模型两大方向:

- 基于不同的卷积计算方式构造新的网络结构
- 在已训练好的模型上做裁剪
  - Pruning：从权重（weight）层面或从（kernel & channel）层面对模型进行修剪；
  - Compressing：如权重共享（clustering based weight sharing）或对权重采用Huffman编码等；
  - Low-bit representing ：如权重量化（quantization）将浮点型转换到[0~255]，或者将网络二值（binary）化等。

 

## All Convolution Net

 All Convolution Net抛弃了以往CNN网络中的池化层和全连接层，通过使用步长更大的卷积层来代替池化以及使用卷积核为1的卷积层来代替全连接层。 

池化层的作用：

1. p-norm（p范数）使CNN的表示更具不变性（invariance）；
2. 降维使高层能够覆盖输入层的更多部分（receptive field）；
3. 池化的feature-wise特性能够使得优化更为容易。 



##   **MobileNet** 

depth-wise separable convolution卷积的方式来代替传统卷积的方式， depth-wise separable convolution包含两种操作：

1. depth-wise convolution，做通道的单独特征计算；
2. point-wise convolution，产生联合特征对DW Convolution生成特征的线性组合。好处是减少参数数量的同时也提升了计算效率。

相较于 GoogLeNet，在同一个量级的参数情况下，但是在运算量上小，同时也保证了较高的准确率。



##  ShuffleNet

point-wise convolution这种操作实际上是非常耗时的，为了能够高效的在输入特征图间建立信息流通，通过利用group convolution和channel shuffle这两个操作来设计卷积神经网络模型，在减少了参数的同时也能够有效提高计算效率。

