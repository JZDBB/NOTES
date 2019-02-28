# http://playground.tensorflow.org
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
eta = 0.1

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

def train_2d(trainer):  # 本函数将保存在 gluonbook 包中方便以后使用。
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1 和 s2 是自变量状态，之后章节会使用。
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

if __name__ == '__main__':
    # 1-D gradient descent
    # res = gd(0.2)
    # trace_show(res)

    # 2-D gradient descent
    # show_trace_2d(f_2d, train_2d(gd_2d))

    # stochastic gradient descent
    show_trace_2d(f_2d, train_2d(sgd_2d))