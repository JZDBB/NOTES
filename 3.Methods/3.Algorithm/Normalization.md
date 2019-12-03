# Normalization



### Batch Normalization

当 BatchSize 太小时效果不佳、对 RNN 等动态网络无法有效应用 BN 等

### Layer Normalization



### Instance Normalization



### Group Normalization







**6. normalization**

虽然在 resnet 里的标配是 BN，在分类任务上表现很好，但是图像生成方面，推荐使用其他 normlization 方法，例如
parameterized 方法有 instance normalization [8]、layer normalization [9] 等，non-parameterized 方法推荐使用 pixel normalization [10]。假如你有选择困难症，那就选择大杂烩的 normalization
方法——switchable normalization [11]。

