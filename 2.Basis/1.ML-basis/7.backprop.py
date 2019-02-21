"""
BP神经网络
"""
import numpy as np
import math
import matplotlib.pyplot as plt

# 函数及求导函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def diff_sigmoid(x):
    fval = sigmoid(x)
    return fval * (1 - fval)

def linear(x):
    return x

def diff_linear(x):
    return np.ones_like(x)

class BP:
    def __init__(self, n_hidden=None, f_hidden='sigmoid', f_output='sigmoid',
                 epsilon=1e-3, max_step=1000, learning_rate=0.1, alpha=0.0):
        self.n_input = None  # 输入层神经元数目
        self.n_hidden = n_hidden  # 隐藏层神经元数目
        self.n_output = None
        self.f_hidden = f_hidden
        self.f_output = f_output
        self.epsilon = epsilon
        self.max_step = max_step
        self.learn_rate = learning_rate  # 学习率
        self.alpha = alpha  # 动量因子

        self.W_in2hide = None  # 输入层到隐藏层权值矩阵
        self.W_hide2out = None  # 隐藏层到输出层权值矩阵
        self.B_in2hide = None  # 输入层到隐藏层阈值
        self.B_hide2out = None  # 隐藏层到输出层阈值
        self.N = None # 输入数据维数

    def init_param(self, X_data, y_data):
        # 初始化
        if len(X_data.shape) == 1:  # 若输入数据为一维数组，则进行转置为n维数组
            X_data = np.transpose([X_data])
        self.N = X_data.shape[0]
        if len(y_data.shape) == 1:
            y_data = np.transpose([y_data])
        self.n_input = X_data.shape[1] # 根据输入和输出的维度生成相应的神经元个数
        self.n_output = y_data.shape[1]
        if self.n_hidden is None:
            self.n_hidden = int(math.ceil(math.sqrt(self.n_input + self.n_output)) + 2)

        self.W_in2hide = np.random.rand(self.n_input, self.n_hidden)
        self.W_hide2out = np.random.rand(self.n_hidden, self.n_output)
        self.B_in2hide = np.random.rand(self.n_hidden)
        self.B_hide2out = np.random.rand(self.n_output)
        return X_data, y_data

    def inspirit(self, name):
        # 获取相应的激励函数
        if name == 'sigmoid':
            return sigmoid
        elif name == 'linear':
            return linear
        else:
            raise ValueError('the function is not supported now')

    def diff_inspirit(self, name):
        # 获取相应的激励函数的导数
        if name == 'sigmoid':
            return diff_sigmoid
        elif name == 'linear':
            return diff_linear
        else:
            raise ValueError('the function is not supported now')

    def forward(self, X_data):
        # 前向传播
        x_hidden_in = X_data @ self.W_in2hide + self.B_in2hide  # n*h
        x_hidden_out = self.inspirit(self.f_hidden)(x_hidden_in)  # n*h
        x_output_in = x_hidden_out @ self.W_hide2out + self.B_hide2out  # n*o
        x_output_out = self.inspirit(self.f_output)(x_output_in)  # n*o
        return x_output_out, x_output_in, x_hidden_out, x_hidden_in

    def fit(self, X_data, y_data):
        # 训练主函数
        X_data, y_data = self.init_param(X_data, y_data)
        step = 0
        # 初始化动量项
        delta_wih = np.zeros_like(self.W_in2hide)
        delta_who = np.zeros_like(self.W_hide2out)
        delta_bih = np.zeros_like(self.B_in2hide)
        delta_bho = np.zeros_like(self.B_hide2out)
        while step < self.max_step:
            step += 1
            # 向前传播
            x_output_out, x_output_in, x_hidden_out, x_hidden_in = self.forward(X_data)
            if np.sum(abs(x_output_out - y_data)) < self.epsilon:
                break
            # 误差反向传播，依据权值逐层计算当层误差
            err_output = y_data - x_output_out  # n*o， 输出层上，每个神经元上的误差
            delta_ho = -err_output * self.diff_inspirit(self.f_output)(x_output_in)  # n*o
            err_hidden = delta_ho @ self.W_hide2out.T  # n*h， 隐藏层（相当于输入层的输出），每个神经元上的误差
            # 隐藏层到输出层权值及阈值更新
            delta_bho = np.sum(self.learn_rate * delta_ho + self.alpha * delta_bho, axis=0) / self.N
            self.B_hide2out -= delta_bho
            delta_who = self.learn_rate * x_hidden_out.T @ delta_ho + self.alpha * delta_who
            self.W_hide2out -= delta_who
            # 输入层到隐藏层权值及阈值的更新
            delta_ih = err_hidden * self.diff_inspirit(self.f_hidden)(x_hidden_in)  # n*h
            delta_bih = np.sum(self.learn_rate * delta_ih + self.alpha * delta_bih, axis=0) / self.N
            self.B_in2hide -= delta_bih
            delta_wih = self.learn_rate * X_data.T @ delta_ih + self.alpha * delta_wih
            self.W_in2hide -= delta_wih
        return

    def predict(self, X):
        # 预测
        res = self.forward(X)
        return res[0]

if __name__ == '__main__':
    N = 50
    X_data = np.linspace(-1, 1, N)
    X_data = np.transpose([X_data])
    y_data = np.exp(-X_data) * np.sin(2 * X_data)
    bp = BP(f_output='linear', max_step=500, learning_rate=0.01, alpha=0.1)  # 注意学习率若过大，将导致不能收敛
    bp.fit(X_data, y_data)
    plt.plot(X_data, y_data)
    pred = bp.predict(X_data)
    plt.scatter(X_data, pred, color='r')
    plt.show()
