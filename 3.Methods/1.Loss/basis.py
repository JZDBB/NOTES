# http://playground.tensorflow.org
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import math

eta = 0.1
eta2 = 0.4

def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x # f(x) = x * x 的导数为 f'(x) = 2 * x。
        results.append(x)
    print('epoch 10, x:', x)
    return results

def trace_show(res):
    res = np.array(res)
    res_y = res ** 2
    x = np.linspace(-12, 12, 1000)
    y = x ** 2
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, color="red", linewidth=2)
    plt.plot(res, res_y, color="blue", marker='o', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("title")
    # plt.ylim([10, 10])
    plt.show()

def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1 和 s2 是自变量状态
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
        print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results

def show_trace_2d(f, results):
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def f_2d(x1, x2):  # 目标函数。
    return x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)

def sgd_2d(x1, x2, s1, s2):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
            x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)

def f_2d2(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d2(x1, x2, s1, s2):
    return (x1 - eta2 * 0.2 * x1, x2 - eta2 * 4 * x2, 0, 0)

eta3, gamma = 0.4, 0.5
def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta3 * 0.2 * x1
    v2 = gamma * v2 + eta3 * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

eta4 = 2
def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # 前两项为自变量梯度。
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta4 / math.sqrt(s1 + eps) * g1
    x2 -= eta4 / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

eta5, gamma2 = 0.4, 0.9
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma2 * s1 + (1 - gamma2) * g1 ** 2
    s2 = gamma2 * s2 + (1 - gamma2) * g2 ** 2
    x1 -= eta5 / math.sqrt(s1 + eps) * g1
    x2 -= eta5 / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2



def network(learning_rate, batch_size):
    pass

if __name__ == '__main__':
    # 1-D gradient descent
    # res = gd(0.2)
    # trace_show(res)

    # 2-D gradient descent
    # show_trace_2d(f_2d, train_2d(gd_2d))

    # stochastic gradient descent
    # show_trace_2d(f_2d, train_2d(sgd_2d))

    # momentum
    # show_trace_2d(f_2d2, train_2d(gd_2d2))
    # show_trace_2d(f_2d, train_2d(momentum_2d))

    # adagrad
    # show_trace_2d(f_2d2, train_2d(adagrad_2d))

    # rmsprop
    show_trace_2d(f_2d2, train_2d(rmsprop_2d))