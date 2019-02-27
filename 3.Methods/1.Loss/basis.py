# http://playground.tensorflow.org
import tensorflow
import matplotlib.pyplot as plt
import numpy as np

def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x # f(x) = x * x 的导数为 f'(x) = 2 * x。
        results.append(x)
    print('epoch 10, x:', x)
    return results

if __name__ == '__main__':
    res = gd(0.2)
    res = np.array(res)
    res_y = res ** 2
    x = np.linspace(-10, 10, 1000)
    y = x ** 2
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label="$x^2$", color="red", linewidth=2)
    plt.plot(res, res_y, label="$data$", color="blue", marker='o', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("title")
    # plt.ylim([10, 10])
    plt.show()