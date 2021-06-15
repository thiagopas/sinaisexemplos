import matplotlib.pyplot as plt
import numpy as np
from funcoes import conv

n = np.arange(-1, 7)
x = np.zeros_like(n, dtype=float)
x[1:5] = [2, 1, 2, 1]
h = np.zeros_like(x)
h[1:4] = [1, 2, 1]
axis = [n[0], n[-1], -.1, 2.1]
y = conv(x, h, n)
y[0:-1] = y[1:]
y[-1] = 0

plt.subplot(2, 1, 1)
plt.title('Entrada e saída')
plt.stem(n, x)
plt.grid()
plt.axis(axis)
plt.subplot(2, 1, 2)
plt.stem(n, h)
plt.grid()
plt.axis(axis)

axis = [n[0], n[-1], -.1, 2.6]
plt.figure()
plt.suptitle('LIT')
plt.subplot(4, 2, 1)
x_ = np.zeros_like(x)
x_[1] = 1
plt.stem(n, x_)
plt.grid()
plt.axis(axis)
plt.subplot(4, 2, 2)
h_ = np.copy(h)
plt.stem(n, h_)
plt.grid()
plt.axis(axis)
plt.subplot(4, 2, 3)
x_ = np.zeros_like(x)
x_[2] = 1
plt.stem(n, x_)
plt.grid()
plt.axis(axis)
plt.subplot(4, 2, 4)
h_ = np.zeros_like(h)
h_[1:] = h[:-1]
plt.stem(n, h_)
plt.grid()
plt.axis(axis)
plt.subplot(4, 2, 5)
x_ = np.zeros_like(x)
x_[2] = .5
plt.stem(n, x_)
plt.grid()
plt.axis(axis)
plt.subplot(4, 2, 6)
h_ = np.zeros_like(h)
h_[2:] = h[:-2] * .5
plt.stem(n, h_)
plt.grid()
plt.axis(axis)
plt.subplot(4, 2, 7)
x_ = np.zeros_like(x)
x_[1:3] = [1, .5]
plt.stem(n, x_)
plt.grid()
plt.axis(axis)
plt.subplot(4, 2, 8)
h_ = np.copy(h)
h_[1:] += h[:-1] * .5
plt.stem(n, h_)
plt.grid()
plt.axis(axis)

plt.figure()
axis = [n[0], n[-1], -.1, 2.1]
plt.subplot(4, 1, 1)
plt.title('δ[n-k]x[k]')
for i in np.arange(1, 5):
    plt.subplot(4, 1, i)
    x_ = np.zeros_like(x)
    x_[i] = x[i]
    plt.stem(n, x_)
    plt.grid()
    plt.axis(axis)

plt.figure()
axis = [n[0], n[-1], -.1, 4.1]
plt.title('h[n-k]x[k]')
for i in np.arange(1, 5):
    plt.subplot(4, 1, i)
    y_ = np.zeros_like(x)
    y_[i:i+3] = h[1:4] * x[i]
    plt.stem(n, y_)
    plt.grid()
    plt.axis(axis)

plt.figure()
plt.stem(n, y)
plt.grid()
axis = [n[0], n[-1], -.1, 6.1]
plt.axis(axis)
