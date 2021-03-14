from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate, impulse
import numpy as np
import matplotlib.pyplot as plt


def pixaxis(ax):
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels(["{:.0f}".format(x / np.pi) + 'π' for x in ticks_loc])

# Questão 1a
sp = plt.subplots(1, 2)
ax_x = sp[0].axes[0]
ax_y = sp[0].axes[1]
t = np.arange(-3, 16, .01)
xtick = np.arange(-3, 16)
axis = [-1, 15, -.5, 10.2]
x = u(t)
h = 2*u(t) - u(t-3) - u(t-6)
y = conv(x, h, t)
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick)
plotit(x=t, y=y, ax=ax_y, axis=axis, wait=False, save=False, xticks=xtick)

sp = plt.subplots(1, 1)
ax = sp[0].axes[0]
plotit(x=t, y=y, ax=ax, axis=[-1, 15, -1, 10], xlabel='t [s]', ylabel='y(t)', xticks=xtick, linewidth=3)

# Questão 1b
x = impulse(t)+impulse(t-1)+impulse(t-2)
y = conv(x, h, t)
sp = plt.subplots(1, 2)
ax_x = sp[0].axes[0]
ax_y = sp[0].axes[1]
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick)
plotit(x=t, y=y, ax=ax_y, axis=axis, wait=False, save=False, xticks=xtick)

sp = plt.subplots(1, 1)
ax = sp[0].axes[0]
plotit(x=t, y=y, ax=ax, axis=[-1, 15, -1, 7], xlabel='t [s]', ylabel='y(t)', xticks=xtick, linewidth=3)


# Questão 1c
sp = plt.subplots(1, 1)
ax = sp[0].axes[0]
t = np.arange(-25, 25, .01)
xtick = np.arange(-24, 24, 6)
def h_def(t):
    return 2*u(t) - u(t-3) - u(t-6)
w = np.zeros_like(t)
for k in np.arange(-10, 10):
    w += h_def(t - 12*k) - h_def(t-6-12*k)
plotit(x=t, y=w, ax=ax, axis=[-20, 20, -2.2, 2.2], xlabel='t [s]', xticks=xtick, ylabel='w(t)', linewidth=3)


def x_def(t):
    return (1 - 1e-3/np.pi * t) * (u(t)-u(t-1e3*np.pi))

t = np.arange(-2e3*np.pi, 2e3*np.pi)
xtick = np.arange(-2e3*np.pi, 2.1e3*np.pi, 500*np.pi)
axis = [-2e3*np.pi, 2e3*np.pi, -.1, 1.1]
x = x_def(t) + x_def(-t)
sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
plotit(x=t, y=x, ax=ax_x, axis=axis, xticks=xtick, linewidth=3, xlabel='ω[rad/s]', title='H(ω)')
pixaxis(ax_x)

