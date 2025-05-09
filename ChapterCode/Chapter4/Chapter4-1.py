# 第4章 神经网络

## 4.1 神经网络（Neural Network）

### 4.1.2 激活函数

### 阶跃函数sign(x)
import numpy as np
import matplotlib.pylab as plt

def sign(x):
    return np.array(x > 0, dtype=np.int32)

def grad_sign(x):
    return np.zeros_like(x)

x = np.arange(-5.0,5.0, 0.1)
plt.ylim(-0.1, 1.1) # 指定y轴的范围
plt.plot(x, sign(x),label="sigmoid")
plt.plot(x, grad_sign(x),label="derivative")
plt.legend(loc="upper right", frameon=False)
plt.show()