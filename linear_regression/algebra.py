#!/usr/local/bin/python3

# 1.导入数据集
import inline as inline
import matplotlib
import numpy as np

x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])
y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

# 2.显示数据到坐标轴上
from matplotlib import pyplot as plt
# %matplotlib inline

plt.scatter(x,y)
plt.xlabel("Area")
plt.ylabel('Price')
plt.show()

# 3.定义拟合直线
def f(w1, b, x):
    y = b + w1 * x
    return y

# 3.2 平方损失函数
def loss_square(x, y, b, w):
    loss = sum(np.square(y - (b + w * x)))
    return loss

# 3.3 平方损失函数最小时对应的w，b值
def calculator(x, y):
    n = len(x)
    w1 = (n * sum(x * y) - sum(x) * sum(y)) / (n * sum(x * x) - sum(x) * sum(x))
    w0 = (sum(x * x) * sum(y) - sum(x) * sum(x * y)) / (n * sum(x * x) - sum(x) * sum(x))
    return w0, w1

# 3.4 代入计算
calculator(x, y)

b = calculator(x, y)[0]
w = calculator(x, y)[1]

loss_square(x, y, b, w)

# 4.绘制图像
x_tmp = np.linspace(50, 120, 100)
plt.scatter(x, y)
plt.scatter(x_tmp, b + w * x_tmp, c='r')
plt.show()

# 5.如果手里有100平房子要出售，预估价格
print(f(100, b, w))