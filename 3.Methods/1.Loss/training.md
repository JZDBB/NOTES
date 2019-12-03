# train loss与test loss结果分析

train loss 不断下降，test loss不断下降，说明网络仍在学习;
train loss 不断下降，test loss趋于不变，说明网络过拟合;
train loss 趋于不变，test loss不断下降，说明数据集100%有问题;
train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目;
train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题。







# GAN loss

为什么GAN的Loss一直降不下去。GAN到底什么时候才算收敛？其实，作为一个训练良好的GAN，其Loss就是降不下去的。衡量GAN是否训练好了，只能由人肉眼去看生成的图片质量是否好。不过，对于没有一个很好的评价是否收敛指标的问题，也有许多学者做了一些研究，后文提及的WGAN就提出了一种新的Loss设计方式，较好的解决了难以判断收敛性的问题。下面我们分析一下GAN的Loss为什么降不下去？

生成器和判别器的目的相反，也就是说两个生成器网络和判别器网络互为对抗，此消彼长。不可能Loss一直降到一个收敛的状态。

对于生成器，其Loss下降快，很有可能是判别器太弱，导致生成器很轻易的就"愚弄"了判别器。

对于判别器，其Loss下降快，意味着判别器很强，判别器很强则说明生成器生成的图像不够逼真，才使得判别器轻易判别，导致Loss下降很快。

也就是说，无论是判别器，还是生成器。loss的高低不能代表生成器的好坏。一个好的GAN网络，其GAN Loss往往是不断波动的。

```python
scheme_index = ii//1000 if ii < 10000 else -1
if train_d:
  real_score,fake_score,_,dLoss,gLoss = sess.run([m_real_score,m_fake_score,d_trainer,d_loss,g_loss],
                feed_dict={real_image:rib, inptG:nb,
  gn_stddev:stddev_scheme[scheme_index], training:True})
else:
  real_score,fake_score,_,dLoss,gLoss = sess.run([m_real_score,m_fake_score,g_trainer,d_loss,g_loss],
                feed_dict={real_image:rib, inptG:nb,
  gn_stddev:stddev_scheme[scheme_index], training:True})
if dLoss-gLoss > switch_threshold or real_score < real_score_threshold: 
  train_d = True
else: train_d = False
```

