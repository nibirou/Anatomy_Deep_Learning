# 第4章 神经网络

## 4.1 神经网络（Neural Network）

### 4.1.2 激活函数

### 4. LeakRelu函数
import numpy as np
import matplotlib.pylab as plt

def leakRelu(x,k=0.2):
    y = np.copy( x )
    y[ y < 0 ] *= k        
    return y

def grad_leakRelu(x,k=0.2):
    return np.clip(x > 0, k, 1.0)
    grad = np.ones_like(x)
    grad[x < 0] = alpha
    return grad
  
x = np.arange(-5.0, 5.0, 0.1)
plt.plot(x, leakRelu(x),label="leakrelu")
plt.plot(x, grad_leakRelu(x),label="derivative")
plt.legend(loc="upper right", frameon=False)
plt.show()