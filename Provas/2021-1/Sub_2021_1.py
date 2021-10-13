from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate, exp_u, sinc
import numpy as np
import matplotlib.pyplot as plt

def square(t, t0, t_init, min, max, duty):
    t_per = np.mod(t-t_init, t0)/t0
    x = np.zeros_like(t)
    x[np.where(np.abs(t_per) <= duty)] = max
    x[np.where(np.abs(t_per) > duty)] = min
    return x

def pixaxis(ax):
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels(["{:.0f}".format(x / np.pi) + 'π' for x in ticks_loc])

sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
w = np.arange(-.1, .1, 1e-5)
axis = [-.02, .08, -.1, 1.1]
# Onda quadrada
x = square(t, .005, -.0025/2, 0., 1., .5)
plotit(ax=ax_x, x=t, y=x, axis=axis, axes=True, title='x(t)', save=False)