# 第4章 神经网络

## 4.1 神经网络（Neural Network）

### 4.1.6 损失函数
# 无论是训练神经网络，还是使用训练好的神经网络进行预测，都需要对预测值（输出值）
# 和真实值之间进行误差评估，误差也被称为损失或代价

# 均方差损失函数
import numpy as np

# 单个样本
f = np.array([0.1, 0.2,0.5])
y = np.array([0.3, 0.4,0.2])
loss =  np.sum((f - y) ** 2)/2 
print(loss)

# 多个样本
F = np.array([[0.1, 0.2,0.5],[0.1, 0.2,0.5]])
Y = np.array([[0.3, 0.4,0.2],[0.3, 0.4,0.2]])

m = F.shape[0] #len(F)
loss =  np.sum((F - Y) ** 2)/(2*m)
# loss = (np.square(H-Y)).mean() 
print(loss)

# 均方差函数
def mse_loss(F,Y,divid_2=False):
    m = F.shape[0]
    loss =  np.sum((F - Y) ** 2)/m
    if divid_2:
        loss/=2
    return loss

mse_loss(F,Y,True)