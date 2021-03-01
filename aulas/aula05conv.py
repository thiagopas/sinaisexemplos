from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate, impulse
import numpy as np
import matplotlib.pyplot as plt

def x_RLi(t, R, L):
    return np.exp(-t*R/L)*u(t) / L

sp = plt.subplots(1, 2)
ax_x = sp[0].axes[0]
ax_y = sp[0].axes[1]
t = np.arange(-3, 10, .01)
axis = [-1, 5, -.5, 2]
x = impulse(t-1)
h = x_RLi(t, R=2, L=1)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, dirac=[[0, 1], [.5, 1]])
plotit(x=t, y=y, ax=ax_y, axis=axis)
