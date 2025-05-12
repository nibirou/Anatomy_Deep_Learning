# 第4章 神经网络

## 4.1 神经网络（Neural Network）

### 4.1.2 激活函数

### 3. ReLU函数
import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0, x)
def grad_relu(x):
    return 1. * (x > 0)

x = np.arange(-5.0, 5.0, 0.1)
plt.plot(x, relu(x),label="relu")
plt.plot(x, grad_relu(x),label="derivative")
plt.legend(loc="upper right", frameon=False)
plt.show()