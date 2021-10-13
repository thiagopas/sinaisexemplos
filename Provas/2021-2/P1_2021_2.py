from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate, impulse
import numpy as np
import matplotlib.pyplot as plt


def pixaxis(ax):
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels(["{:.0f}".format(x / np.pi) + 'π' for x in ticks_loc])


# Questão 1
sp = plt.subplots(1)
ax_x = sp[0].axes[0]
t = np.arange(-3, 6, .01)
xtick = np.arange(-3, 7)
axis = [-1, 6, -.2, 2.2]
h = t * (u(t)-u(t-2))
plotit(x=t, y=h, ax=ax_x, axis=axis, xticks=xtick, yticks=[0, 1, 2], xlabel='t [s]', ylabel='h(t)', linewidth=3)

# Questão 1a
sp = plt.subplots(1)
ax_x = sp[0].axes[0]
x = u(t)
y = conv(x, h, t)
plotit(x=t, y=y, ax=ax_x, axis=axis, xticks=xtick, xlabel='t [s]', ylabel='y(t)', linewidth=3)

# Questão 1b
sp = plt.subplots(1)
ax_x = sp[0].axes[0]
x = u(t) - u(t-3)
y = conv(x, h, t)
plotit(x=t, y=y, ax=ax_x, axis=axis, xticks=xtick, xlabel='t [s]', ylabel='y(t)', linewidth=3)

# Questão 2
sp = plt.subplots(1, 2)
ax_x = sp[0].axes[0]
ax_h = sp[0].axes[1]
t = np.arange(-.15, .16, .0001)
xtick = np.arange(-.15, .16, .05)
axis = [-.15, .15, -.1, 1.1]
x = np.array(np.mod(t-.025, .1) >= .05, dtype=float)
h = 2*u(t) - u(t-3) - u(t-6)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick, xlabel='t [s]', title='x(t)', linewidth=3)

def x_def(t):
    return 1 + (u(t+110*np.pi)-u(t-110*np.pi)) * np.abs(t)/(100*np.pi)

t = np.arange(-200*np.pi, 200*np.pi, .1)
xtick = np.arange(-100*np.pi, 150*np.pi, 50*np.pi)
axis = [-120*np.pi, 120*np.pi, -.2, 2.2]
h = x_def(-t)
plotit(x=t, y=h, ax=ax_h, axis=axis, xticks=xtick, linewidth=3, xlabel='ω[rad/s]', title='H(ω)')
pixaxis(ax_h)