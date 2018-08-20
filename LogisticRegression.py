# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:33:55 2018

@author: Administrator
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # 利用ListedColormap设置 marker generator 和 color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 确定横纵轴边界
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 最小-1, 最大+1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # 建立一对grid arrays(网格阵列)
    # 铺平grid arrays，然后进行预测
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) # .revel()降维函数
    Z = Z.reshape(xx1.shape)
    
    # 将不同的决策边界对应不同的颜色
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # 设置坐标轴的范围
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # 绘制样本点
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


class LogisticRegression(object):
    """
    参数
    ----------
    eta : float
        学习速率 (between 0.0 and 1.0)
    n_iter : int
        迭代次数
    -----------
    属性
    ----------
    w_ : 1d-array
        拟合后的权重, 即θ
    cost_ : list
        每次迭代的代价, 即Jθ
    """
    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter

    # 训练函数
    def fit(self, X, y):
        
        """ 
        参数
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            训练集特征向量, n_samples为样本数量, n_features为特征向量的数量
        y : array-like, shape = [n_samples]
            训练集的目标值
        ----------
        Returns
        ----------
        self : object
        """
        
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            y_val = self.activation(X)
            errors = (y - y_val)
            neg_grad = X.T.dot(errors)
            self.w_[1:] += self.eta * neg_grad
#            self.w_[0] += self.eta * errors.sum()
            self.cost_.append(self._logit_cost(y, self.activation(X)))
        return self

    def _logit_cost(self, y, y_val):
        """计算代价函数值"""
        logit = -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))
        return logit
    
    def _sigmoid(self, z):
        """计算逻辑函数值"""
        return 1.0 / (1.0 + np.exp(-z))
    
    def net_input(self, X):
        """计算逻辑函数输入"""
        return np.dot(X, self.w_[1:]) # + self.w_[0]

    def activation(self, X):
        """激活逻辑神经元"""
        z = self.net_input(X)
        return self._sigmoid(z)
    
    def predict_proba(self, X):
        """样本X为1的概率的估计值"""
        return self.activation(X)

    def predict(self, X):
        """预测X的标签, 将大于0的值归为1, 小于0的归为0"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
    
os.chdir('C:/Users/Administrator/Desktop/jpynb/机器学习')

df = pd.read_csv('data/iris.csv')

# select setosa and versicolor
# 两种各选择50个, 把类别改为 0 和 1, 方便画图
y = df.iloc[0:100, 4].values
y = np.where(y == 'setosa', 0, 1)

# 提取 sepal length 和 petal length 两种特征的数据
X = df.iloc[0:100, [0, 2]].values

# 特征标准化
X_std = np.copy(X)
sc_x = StandardScaler()
X_std = sc_x.fit_transform(X)

# 绘制代价函数
lr = LogisticRegression(n_iter=500, eta=0.02).fit(X_std, y)
plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Logistic Regression - Learning rate 0.02')
plt.tight_layout()
plt.savefig('C:/Users/Administrator/Desktop/cost.png', dpi=300)

# 绘制分类边界
plot_decision_regions(X_std, y, classifier=lr)
plt.title('Logistic Regression - Gradient Descent')
plt.xlabel('sepal length [std]')
plt.ylabel('petal length [std]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('C:/Users/Administrator/Desktop/cost.png', dpi=300)
