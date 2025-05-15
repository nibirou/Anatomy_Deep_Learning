# 第6章 卷积神经网络CNN
## 6.1 卷积
### 6.1.2 same卷积与full卷积
import numpy as np

def conv1d(x,w,pad):
    n = x.size
    K = w.size
    P = 2*pad
    n_o = n+P-K+1
    y = np.zeros(n_o)
    if P>0:
        x_pad = np.zeros(n+P)   
        x_pad[pad:-pad] = x
    else: 
        x_pad = x   
    
    for i in range(n_o): 
        y[i] = np.sum(x_pad[i:i+K]*w)
    return y

