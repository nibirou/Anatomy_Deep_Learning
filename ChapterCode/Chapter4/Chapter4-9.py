# 第4章 神经网络

## 4.1 神经网络（Neural Network）

### 4.1.6 损失函数
# 无论是训练神经网络，还是使用训练好的神经网络进行预测，都需要对预测值（输出值）
# 和真实值之间进行误差评估，误差也被称为损失或代价

# 多分类交叉熵损失函数

import numpy as np

def cross_entropy_loss_onehot(F,Y):
    m = len(F)  #F.shape[0] 
    return -(1./m) *np.sum(np.multiply(Y, np.log(F)))

F = np.array([[0.2,0.5,0.3],[0.4,0.3,0.3]])
Y = np.array([[0,0,1],[1,0,0]])
cross_entropy_loss_onehot(F,Y)

def cross_entropy_loss(F,Y,onehot=False):
    m = len(F) #F.shape[0]      #样本数
    if onehot:
        return -(1./m) *np.sum(np.multiply(Y, np.log(F)))
    else: return  - (1./m) *np.sum( np.log(F[range(m),Y]) )  # F[i]中对应Y[i]的那个分类的log值   

F = np.array([[0.2,0.5,0.3],[0.4,0.3,0.3]])  #每行对应一个样本
Y = np.array([2,0])  #第1个样本属于第2类、第2个样本属于第0类，主要是Y的表示方法有所不同

cross_entropy_loss(F,Y)

# 下面代码将整数索引数组转换为一个one-hot数组
n_C = np.max(Y) + 1
one_hot_y = np.eye(n_C)[Y]
cross_entropy_loss_onehot(F, one_hot_y)