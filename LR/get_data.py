import numpy as np
import matplotlib.pyplot as plt
from LR import LogisticRegression
# 设置随机种子，以便结果可复现
np.random.seed(42)

# 生成随机数据
# 两个特征的均值和方差
mean_1 = [2, 2]
cov_1 = [[2, 0], [0, 2]]
mean_2 = [-2, -2]
cov_2 = [[1, 0], [0, 1]]

# 生成类别1的样本
X1 = np.random.multivariate_normal(mean_1, cov_1, 50)
y1 = np.zeros(50)

# 生成类别2的样本
X2 = np.random.multivariate_normal(mean_2, cov_2, 50)
y2 = np.ones(50)

# 合并样本和标签
X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2))

# 绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Dataset')
plt.show()



logreg = LogisticRegression()

# 训练模型
logreg.fit(X, y)

# 预测样本
X_new = np.array([[2.5, 2.5], [-6.0, -4.0]])
y_pred_prob = logreg.predict_prob(X_new)
y_pred = logreg.predict(X_new)

print("Predicted Probabilities:", y_pred_prob)
print("Predicted Labels:", y_pred)