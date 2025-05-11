# 第4章 神经网络

## 4.1 神经网络（Neural Network）

### 4.1.6 损失函数
# 无论是训练神经网络，还是使用训练好的神经网络进行预测，都需要对预测值（输出值）
# 和真实值之间进行误差评估，误差也被称为损失或代价

# 二分类交叉熵损失函数

import numpy as np

f = np.array([0.1, 0.2,0.5])   #3个样本对应分类1的概率
y = np.array([0,   1,   0])   #3个样本对应的分类
m = y.shape[0]

loss = - (1./m)*np.sum(np.multiply(y,np.log(f)) + np.multiply((1 - y), np.log(1 - f)))

print(loss)