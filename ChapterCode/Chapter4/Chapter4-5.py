# 第4章 神经网络

## 4.1 神经网络（Neural Network）

### 4.1.3 神经网络的正向计算（激活函数为sigmoid）

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

g1 = sigmoid
g2 = sigmoid

# x和W1,b1
x = np.array([1.0, 0.5])              # 输入x： 1x2行向量
W1 = np.array([[0.1, 0.3,0.5,0.2],
               [0.4,0.6,0.7, 0.1]])   # W1 ： 2x4矩阵
b1 = np.array([0.1, 0.2, 0.3,0.4])    # 偏置b1： 1x4行向量
print("x.shape",x.shape)                        # (2,)
print("W1.shape",W1.shape)                       # (2, 4)
print("b1.shape",b1.shape)                       # (4,)

# 从输入x和W1,b1计算z1和a1的值
z1 = np.dot(x,W1) + b1                # (1,4)
a1 = g1(z1)                      # (1,4)
print("z1",z1)                             # (4,)
print("a1",a1) 

# a1、W2,b2
W2 = np.array([[0.1, 1.4,0.2],[2.5, 0.6, 0.3],[1.1,0.7,0.8],[0.3,1.5,2.1]])
b2 = np.array([0.1, 2,0.3])
print("a2.shape",a1.shape) # (4,)
print("W2.shape",W2.shape) # (2, 4)
print("b2.shape",b2.shape) # (2,)

# 从a1、W2,b2计算z2和a2的值
z2 = np.dot(a1,W2) + b2
a2 = g2(z2)
print("z2",z2)
print("a2",a2)