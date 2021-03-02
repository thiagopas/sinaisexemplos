from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate, impulse
import numpy as np
import matplotlib.pyplot as plt


def gaussian(t, avg, std):
    x = np.exp(-.5*((t-avg)/std)**2)
    thewindow = np.copy(x[np.where(np.array(t >= 3, dtype=int) * np.array(t <= 3.7, dtype=int))])
    line = np.zeros_like(thewindow)
    step = (1.5 - thewindow[0]) / len(thewindow)
    line = np.arange(thewindow[0], 1.5, step)
    x[np.where(np.array(t >= 3, dtype=int) * np.array(t <= 3.7, dtype=int))] = line
    return x

# Não normalizado
def r_(t, width):
    #x = (t >= -width/2) * (t < width/2) + 0.0000
    x = (t >= 0) * (t < width) + 0.0000
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

sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
t = np.arange(-3, 10, .01)
xtick = np.arange(-3, 10)
axis = [-1, 5, -.2, 1.6]
x = gaussian(t, 2, 1.5)
h = x_RLi(t, 2, 1)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False, wait=True)
x_ = approx(x, t, 1)
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick, linewidth=2)
plotit(x=t, y=x_, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False, hold=True, color='C1')
x_ = approx(x, t, .5)
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick, linewidth=2)
plotit(x=t, y=x_, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False, hold=True, color='C1')
x_ = approx(x, t, .25)
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick, linewidth=2)
plotit(x=t, y=x_, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False, hold=True, color='C1')
x_ = approx(x, t, .1)
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick, linewidth=2)
plotit(x=t, y=x_, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False, hold=True, color='C1')

r = (u(t) - u(t-.66))/.66
plotit(x=t, y=r, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False)
h = x_RLi(t, 2, 1)
y = conv(r, h, t)
plotit(x=t, y=y, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False)
x_ = approx(x, t, .66)
plotit(x=t, y=x_, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False)
y = conv(x_, h, t)
plotit(x=t, y=y, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False)

x = impulse(t)
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, dirac=[[0], [1]], save=False)
y = conv(x, h, t)
plotit(x=t, y=y, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False)
x = gaussian(t, 2, 1.5)
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False)
y = conv(x, h, t)
plotit(x=t, y=y, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False)

plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False)
plotit(x=t, y=x_, ax=ax_x, axis=axis, xticks=xtick, linewidth=2, save=False, color='orange')

plotit(x=t, y=r, ax=ax_x, axis=[-1, 2, -.2, 1.6], xticks=xtick, linewidth=3, save=True, color='orange')