import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 目标函数
def real_func(x):
    return np.sin(2*np.pi*x)

# 多项式
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret

def fitting(x, y, M=0):
    """
    M    为 多项式的次数
    """
    # 随机初始化多项式参数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    print('Fitting Parameters:', p_lsq[0])
    return p_lsq

def residuals_func_regularization(p, x, y):
    regularization = 0.0001
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(0.5*regularization*np.square(p))) # L2范数作为正则化项
    return ret

def regular_LSM(x, y, M=0):
    # 最小二乘法,加正则化项
    p_init = np.random.rand(M+1)
    p_lsq_regularization = leastsq(residuals_func_regularization, p_init, args=(x, y))

    return p_lsq_regularization


def main():
    # data
    x = np.linspace(0, 1, 10)
    x_points = np.linspace(0, 1, 1000)
    # 加上正态分布噪音的目标函数的值
    y_ = real_func(x)
    y = [np.random.normal(0, 0.1) + y1 for y1 in y_]
    M = 9

    p_lsq_0 = fitting(x, y, M=M)
    p_lsq_regularization = regular_LSM(x, y, M=M)

    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq_0[0], x_points), label='fitted curve')
    plt.plot(x_points, fit_func(p_lsq_regularization[0], x_points), label='fitted_regular curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    main()
