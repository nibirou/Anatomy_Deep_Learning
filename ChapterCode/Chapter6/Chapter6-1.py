# 第6章 卷积神经网络CNN
## 6.1 卷积
### 6.1.1 什么是卷积？
# 第四章介绍的全连接神经网络默认输入的所有数据样本都是一维张量，每一层的神经网络都接收来自前一层的一维张量产生的一个输出
# 卷积神经
import numpy as np
np.random.seed(5)
x = np.random.randint(low=1, high=30, size=10,dtype='l')
print(x)

w = np.array([1.2,0.3,0.5])
n = x.size
K = w.size
z = np.zeros(n-K+1)
for i in range(n-K+1):   
    z[i] = np.sum(x[i:i+K]*w)
print(w)
print(z)