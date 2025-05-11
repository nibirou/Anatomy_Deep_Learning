# 第4章 神经网络

## 4.1 神经网络（Neural Network）

### 4.1.4 多个样本的正向计算（激活函数为sigmoid）

### m个样本 xi  组成一个矩阵

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([[1.0, 2.],[3.0,4.0]])
W1 = np.array([[0.1, 0.3,0.5,0.2],
               [0.4,0.6,0.7, 0.1]])   # W1 ： 2x4矩阵
b1 = np.array([0.1, 0.2, 0.3,0.4])    # 偏置b1： 1x4行向量

print("X.shape",X.shape) # (2,2)
print("W1.shape",W1.shape) # (2, 4)
print("b1.shape",b1.shape) # (4,)

# 计算第1层的Z1,A1
Z1 = np.dot(X,W1) + b1
A1 = sigmoid(Z1)
print("Z1:",Z1)
print("A1:",A1)

W2 = np.array([[0.1, 1.4,0.2],[2.5, 0.6, 0.3],[1.1,0.7,0.8],[0.3,1.5,2.1]])
b2 = np.array([0.1, 2,0.3])
print("A1.shape",A1.shape) # (2,)
print("W2.shape",W2.shape) # (4, 2)
print("b2.shape",b2.shape) # (4,)

# 计算第1层的Z2,A2
Z2 = np.dot(A1,W2) + b2
A2 = sigmoid(Z2)
print("Z2:",Z2)
print("A2:",A2)
