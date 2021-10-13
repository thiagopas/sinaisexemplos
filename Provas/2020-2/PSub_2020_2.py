import numpy as np
import matplotlib.pyplot as plt
from funcoes import conv_disc

def ustep(t):
    result = (t >= 0)
    return np.array(result, dtype=float)

t = np.arange(-10, 12)
x = np.cos(0.5*np.pi*t)*ustep(t)
h = np.zeros_like(t)
h[t==0] = 1
h[t==1] = 1
h[t==2] = 1
h[t==3] = 0
y = conv_disc(x, h, t)

plt.figure()
plt.stem(t, y)
plt.xticks(t)
plt.grid()
plt.legend(['y[n]'])
plt.xlabel('n')
plt.axis([-2, t.max(), -1.4, 1.4])
