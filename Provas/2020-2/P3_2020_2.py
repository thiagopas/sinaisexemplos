import numpy as np
import matplotlib.pyplot as plt
from funcoes import u, conv_disc

t = np.arange(-2, 12)
x = (u(t) - u(t-4)) * t + (u(t-4) - u(t-8)) * (8-t)
h = np.zeros_like(t)
h[t==0] = 1
h[t==1] = -1
y = conv_disc(x, h, t)

plt.figure()
plt.subplot(1, 2, 1)
plt.stem(t, x)
plt.xticks(t)
plt.grid()
plt.legend(['x[n]'])
plt.xlabel('n')
plt.axis([t.min(), t.max(), -.5, 4.5])

plt.subplot(1, 2, 2)
plt.stem(t, h)
plt.xticks(t)
plt.grid()
plt.legend(['h[n]'])
plt.xlabel('n')
plt.axis([t.min(), t.max(), -1.5, 1.5])

plt.figure()
plt.stem(t, y)
plt.xticks(t)
plt.grid()
plt.legend(['y[n]'])
plt.xlabel('n')
plt.axis([t.min(), t.max(), -1.5, 1.5])
