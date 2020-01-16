from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt #画图

# 1. 定义数据集
x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])
y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

# 1.1 展示数据集合
plt.scatter(x,y)
def runplt(size=None):
    #plt.figure(size)
    plt.title("Area and House Price Relationship")
    plt.xlabel("House Area")
    plt.ylabel("House Price")
    #plt.axis([0,25,0,25])
    plt.grid(True) # 网格
    return plt
plt = runplt()
plt.show()

# 2. 定义线性回归模型
model = LinearRegression()
'''
调用sklearn.linear_model.LinearRegression()所需参数：
* fit_intercept : 布尔型参数，表示是否计算该模型截距。可选参数。
* normalize : 布尔型参数，若为True，则X在回归前进行归一化。可选参数。默认值为False。
* copy_X : 布尔型参数，若为True，则X将被复制；否则将被覆盖。 可选参数。默认值为True。
* n_jobs : 整型参数，表示用于计算的作业数量；若为-1，则用所有的CPU。可选参数。默认值为1。

线性回归fit函数用于拟合输入输出数据，调用形式为model.fit(X,y, sample_weight=None)：
• X : X为训练向量；
• y : y为相对于X的目标向量；
• sample_weight : 分配给各个样本的权重数组，一般不需要使用，可省略。
注意：X，y 以及model.fit()返回的值都是2-D数组，如：a= [ [ 0] ]
'''
# 训练, reshape 操作把数据处理成 fit 能接受的形状
model.fit(x.reshape(len(x), 1), y)

# 3.得到模型拟合参数
model.intercept_, model.coef_

# 4. 预测
print(model.predict([[150]]))

# 5.print model parameter
print(model.intercept_)  #截距
print(model.coef_)  #线性模型的系数

plt.scatter(x, y, c='red') # 展示原始数据点
y2 = model.predict(x.reshape(len(x),1))
plt.plot(x, y2, 'g-') # 展示模型结果点
plt.show()