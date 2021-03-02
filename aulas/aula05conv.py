from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate, impulse
import numpy as np
import matplotlib.pyplot as plt


def gaussian(t, avg, std):
    x = np.exp(-.5*((t-avg)/std)**2)
    return x

# Não normalizado
def r_(t, width):
    x = (t >= -width/2) * (t < width/2) + 0.0000
    return x

def approx(x, t, width):
    t0 = np.argmin(np.abs(t))
    t_s = t[1] - t[0]
    width_s = int(np.round(width / t_s))
    x_hat = r_(t, width) * x[t0]
    if t[0] < 0:
        negative_samples = int(np.floor(-t[0] / width))
        for i in range(1, negative_samples):
            x_hat = x_hat + r_(t + i*width, width) * x[t0 - i*width_s]
    if t[-1] > 0:
        positive_samples = int(np.floor(t[-1] / width))
        for i in range(1, positive_samples):
            x_hat = x_hat + r_(t - i*width, width) * x[t0 + i*width_s]
    return x_hat

def x_RLi(t, R, L):
    return np.exp(-t*R/L)*u(t) / L

sp = plt.subplots(1, 2)
ax_x = sp[0].axes[0]
ax_y = sp[0].axes[1]
t = np.arange(-3, 10, .01)
xtick = np.arange(-3, 10)
axis = [-1, 5, -.5, 2]
h = impulse(t) / 2
x = u(t)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick)
plotit(x=t, y=y, ax=ax_y, axis=axis, wait=True, save=True, xticks=xtick)
x = impulse(t)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, dirac=[[0], [1]], xticks=xtick)
plotit(x=t, y=y, ax=ax_y, axis=axis, save=True, dirac=[[0], [.5]], xticks=xtick)
x = impulse(t) + 2 * impulse(t - 1.5) - .5*impulse(t - 3.)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, dirac=[[0, 1.5, 3.0], [1, 2, -.5]], xticks=xtick)
plotit(x=t, y=y, ax=ax_y, axis=axis, save=True, dirac=[[0, 1.5, 3], [.5, .75, -.25]], xticks=xtick)

h = x_RLi(t, R=2, L=1)
x = u(t)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick)
plotit(x=t, y=y, ax=ax_y, axis=axis,  save=True, xticks=xtick)
x = impulse(t)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, dirac=[[0], [1]], xticks=xtick)
plotit(x=t, y=y, ax=ax_y, axis=axis, save=True, xticks=xtick)
x = impulse(t) + 2 * impulse(t - 1.5) - .5*impulse(t - 3.)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, dirac=[[0, 1.5, 3.0], [1, 2, -.5]], xticks=xtick)
plotit(x=t, y=y, ax=ax_y, axis=axis, save=True, xticks=xtick)

h = .5 * (u(t) - u(t-2))
x = u(t)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick)
plotit(x=t, y=y, ax=ax_y, axis=axis,  save=True, xticks=xtick)
x = impulse(t)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, dirac=[[0], [1]], xticks=xtick)
plotit(x=t, y=y, ax=ax_y, axis=axis, save=True, xticks=xtick)
x = impulse(t) + 2 * impulse(t - 1.5) - .5*impulse(t - 3.)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, dirac=[[0, 1.5, 3.0], [1, 2, -.5]], xticks=xtick)
plotit(x=t, y=y, ax=ax_y, axis=axis, save=True, xticks=xtick)

