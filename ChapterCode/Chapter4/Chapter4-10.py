# 第4章 神经网络

## 4.1 神经网络（Neural Network）

### 4.1.7 基于数值梯度的神经网络训练

### 和回归问题一样，一个结构确定的神经网络函数

import numpy as np

def initialize_parameters(n_x, n_h, n_o):
    np.random.seed(2)            # 固定种子，使得每次运行这个代码的随机数的值总是同样的   
  
    W1 = np.random.randn(n_x,n_h)* 0.01
    b1 = np.zeros((1,n_h))
    W2 = np.random.randn(n_h,n_o) * 0.01
    b2 = np.zeros((1,n_o))
   
    assert (W1.shape == (n_x, n_h))
    assert (b1.shape == (1, n_h))
    assert (W2.shape == (n_h, n_o))
    assert (b2.shape == (1, n_o))
    
    parameters = [W1,b1,W2,b2]
    return parameters