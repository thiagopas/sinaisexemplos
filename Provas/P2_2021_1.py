from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate, impulse
import numpy as np
import matplotlib.pyplot as plt

def pixaxis(ax):
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels(["{:.0f}".format(x / np.pi) + 'π' for x in ticks_loc])

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


sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
w = np.arange(-150*np.pi, 150*np.pi, .5)
axis = [-150*np.pi, 150*np.pi, -.2, 3.2]
# Onda quadrada
x = 3 * (u(w + 100*np.pi) - u(w + 80*np.pi) + u(w - 80*np.pi) - u(w - 100*np.pi))
plotit(ax=ax_x, x=w, y=x, axis=axis, axes=True,
       xlabel='ω [rad/s]', title='|Y(ω)|',
       save=False, xticks=np.array([-100, -80, 0, 80, 100])*np.pi)
pixaxis(ax_x)
