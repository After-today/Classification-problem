# 绘画分类边界
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
        
    # highlight test samples
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='b', 
                alpha=0.1, linewidth=1, marker='o', 
                s=55, label='test sets')
        
# 加载数据
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Class labels:', np.unique(y))
print(iris.target_names)

# 训练集、测试集划分
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)

# 特征标准化处理
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

#利用logistic regression建模类别概率
# C 正则化系数λ的倒数
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=lr, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('C:/Users/Administrator/Desktop/logistic_regression.png', dpi=300)
lr.predict_proba(X_test_std[0,:].reshape(1,-1))
lr.predict(X_test_std[0,:].reshape(1,-1))

# 手动实现带正则化项的逻辑回归算法
class LogitGD(object):
    def __init__(self, eta=0.01, lamb = 0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.lamb = lamb

    def fit(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors) - self.lamb* self.w_[1:]
#            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0 + self.lamb* np.sum(self.w_[1:]**2)
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) #+ self.w_[0]
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def activation(self, X):
        return self.sigmoid(self.net_input(X))

    def predict(self, X):
        return np.where(self.activation(X) >= 0.5, 1, -1)
