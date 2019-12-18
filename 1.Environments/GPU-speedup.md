# 

## pytorch

显存装不下模型权重+中间变量

优化方法：及时清空中间变量，优化代码，减少batch

显存消耗的幕后黑手其实是神经网络中的中间变量以及使用optimizer算法时产生的巨量的中间参数。

显存占用 = 模型参数 + 计算产生的中间变量

 反向传播时，中间变量+原来保存的中间变量，存储量会翻倍 

占用显存的层一般是：

- 卷积层，通常的conv2d
- 全连接层，也就是Linear层
- BatchNorm层
- Embedding层

而不占用显存的则是：

- 刚才说到的激活层Relu等
- 池化层
- Dropout层

具体计算方式：

- Conv2d(Cin, Cout, K): 参数数目：Cin × Cout × K × K
- Linear(M->N): 参数数目：M×N
- BatchNorm(N): 参数数目： 2N
- Embedding(N,W): 参数数目： N × W



 计算模型权重及中间变量占用大小： 

```python
# 模型显存占用监测函数
# model：输入的模型
# input：实际中需要输入的Tensor变量
# type_size 默认为 4 默认类型为 float32 
 
def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
 
    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)
 
    mods = list(model.modules())
    out_sizes = []
 
    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out
 
    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums
 
 
    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))
```







 占用显存大概分以下几类： 

- 模型中的参数(卷积层或其他有参数的层)
- 模型在计算时产生的中间参数(也就是输入图像在计算时每一层产生的输入和输出)
- backward的时候产生的额外的中间参数
- 优化器在优化时产生的额外的模型参数

优化除了算法层的优化，最基本的优化无非也就一下几点：

- 减少输入图像的尺寸
- 减少batch，减少每次的输入图像数量
- 多使用下采样，池化层
- 一些神经网络层可以进行小优化，利用relu层中设置`inplace`
- 购买显存更大的显卡
- 从深度学习框架上面进行优化



 其中Float32 是在深度学习中最常用的数值类型，称为单精度浮点数，每一个单精度浮点数占用4Byte的显存 



 显存占用 = 模型显存占用 + batch_size × 每个样本的显存占用 

### 方法

- ReLU用`inplace=True`
- 用`eval()`和`with torch.no_grad():`
- 每个batch后将所有参数从GPU中拿出删除
- `torch,backends.cudnn.deterministic=True`建议开着
- 不用`.cpu()`来取出GPU中的参数或图片
- 将不需要的变量在`forward`内全用`x`代替，或者用完后用`del`删除
- pytorch-0.4.0`checkpoint`或者` checkpoint_sequential `牺牲计算速度节省显存

-  **因为每次迭代都会引入点临时变量，会导致训练速度越来越慢，基本呈线性增长。开发人员还不清楚原因，但如果周期性的使用torch.cuda.empty_cache()的话就可以解决这个问题。** 
- 数据并行读取：`train_loader`

```python
# num _workers: CPU使用线程. -般建议这个值填写你机器总共CPU的数量
# pin_ memory:是否先把数据加载到缓存再加载到GPU.如果你用的不是你私人工作电脑,请开启.
# drop_ last: 如果是这是训练使用的dataset,请开启，这样最后- -个 batch如果小于你的
# batch_ size, 会扔掉,这样训练就会更稳定.

data_ loader = data. DataLoader(YOUR_ PYTORCH_ DATASET,
num_ workers=THE_ NUMBER_ OF_ CPU_ I_ HAVE,pin_ memory=True,
drop_ last=True, # Last batch will mess up with batch nor))
```

