# 第4章 神经网络

## 4.1 神经网络（Neural Network）

### 4.1.2 激活函数

### tanh函数
import numpy as np
import matplotlib.pylab as plt

def grad_tanh(x):
    a = np.tanh(x)
    return 1 - a**2

x = np.arange(-5.0, 5.0, 0.1)
plt.plot(x, np.tanh(x),label="tanh")
plt.plot(x, grad_tanh(x),label="derivative")
plt.legend(loc="upper right", frameon=False)
plt.show()