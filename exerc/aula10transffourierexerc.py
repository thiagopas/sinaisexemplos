from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate, exp_u, sinc
import numpy as np
import matplotlib.pyplot as plt
import time

def dn(n, t0):
    n = float(n)
    if n == 0:
        return .5
    else:
        return np.sin(n*.5*np.pi)/(n*np.pi)


def dn_n(nvec, t0):
    d_values = np.zeros_like(nvec, dtype=float)
    for i in range(len(nvec)):
        d_values[i] = dn(nvec[i], t0)
    return d_values

def x_RC(t, R, C):
    return np.exp(-t/(R*C))*u(t) / (R*C)

def w_RC(w, R, C):
    return (1/(R*C)) / (1j*w + 1/(R*C))

def w_RL(w, R, L):
    return (1/L) * 1/(R/L + 1j*w)

def piyaxis(ax):
    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(["{:.1f}".format(y / np.pi) + 'π' for y in ticks_loc])

def tauyaxis(ax):
    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(['$' + "{:.1f}".format(y) + '\\tau$' for y in ticks_loc])

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


# espectro do sinal exemplo
sp = plt.subplots(1, 1)
ax_abs = sp[0].axes[0]
w = np.arange(-4*np.pi, 4.001*np.pi, 1e-3)
h = u(w+2*np.pi) - u(w-2*np.pi)
xtick = np.arange(w[0], w[-1], np.pi)
plotit(ax=ax_abs, x=w, y=h, axis=[w[0], w[-1], -.1, 1.1], axes=False, save=False, title='$H(\omega)$', xlabel='$\omega$', xticks=xtick, wait=True)
pixaxis(ax_abs)