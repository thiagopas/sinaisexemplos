from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate, impulse
import numpy as np
import matplotlib.pyplot as plt

def piyaxis(ax):
    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(["{:.3f}".format(y / np.pi) + 'π' for y in ticks_loc])

def square(t, t0, t_init, min, max, duty):
    t_per = np.mod(t-t_init, t0)/t0
    x = np.zeros_like(t)
    x[np.where(np.abs(t_per) <= duty)] = max
    x[np.where(np.abs(t_per) > duty)] = min
    return x


sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
t = np.arange(-.1, .1, 1e-5)
axis = [-.02, .02, -.02, 1.02]
# Onda quadrada
x = square(t, .01, 0., .4, 1., .5)
plotit(ax=ax_x, x=t, y=x, axis=axis, axes=True, xlabel='t [s]', title='x(t)', save=False)