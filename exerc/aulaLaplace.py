from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate, exp_u, sinc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


# Questao SLIT
sp = plt.subplots(1, 1)
ax_abs = sp[0].axes[0]
t = np.arange(-1, 8, 1e-3)
h = np.exp(-1*t)*u(t) - np.exp(-2*t)*u(t)
plotit(ax=ax_abs, x=t, y=h, axes=True, axis=[t[0], t[-1], -0.1, h.max()*1.1], save=False, title='h(t)', xlabel='t', wait=False)

sp = plt.subplots(1, 1)
ax_abs = sp[0].axes[0]
t = np.arange(-1, 7, 1e-3)
h = .5*np.exp(.1*t)*u(t) - np.exp(-1*t)*u(t)
plotit(ax=ax_abs, x=t, y=h, axes=True, axis=[t[0], t[-1], h.min()*1.1, 1], save=False, title='h(t)', xlabel='t', wait=False)

sp = plt.subplots(1, 1)
ax_abs = sp[0].axes[0]
fig = sp[0].figure
t = np.arange(-1, 6, 1e-3)
h = np.exp(-t)*np.cos(4*t)*u(t)
plotit(ax=ax_abs, x=t, y=h, axes=True, axis=[t[0], t[-1], h.min()*1.1, h.max()*1.1], save=False, title='h(t)', xlabel='t', wait=False)
#fig.set_size_inches([6.4, 4.8])

sp = plt.subplots(1, 1)
ax_abs = sp[0].axes[0]
fig = sp[0].figure
t = np.arange(-1, 6, 1e-3)
h = np.exp(-t)*np.sin(4*t)*u(t)
plotit(ax=ax_abs, x=t, y=h, axes=True, axis=[t[0], t[-1], h.min()*1.1, h.max()*1.1], save=False, title='h(t)', xlabel='t', wait=False)
#fig.set_size_inches([6.4, 4.8])

sp = plt.subplots(1, 1)
ax_abs = sp[0].axes[0]
fig = sp[0].figure
t = np.arange(-1, 6, 1e-3)
h = np.sqrt(2)*np.exp(-t)*np.cos(4*t-np.pi/4)*u(t)
#h = np.exp(-t)*np.sin(4*t)*u(t) + np.exp(-t)*np.cos(4*t)*u(t)
plotit(ax=ax_abs, x=t, y=h, axes=True, axis=[t[0], t[-1], h.min()*1.1, h.max()*1.1], save=False, title='h(t)', xlabel='t', wait=False)
#fig.set_size_inches([6.4, 4.8])

sp = plt.subplots(1, 1)
ax_abs = sp[0].axes[0]
fig = sp[0].figure
t = np.arange(-1, 10, 1e-3)
h = np.exp(.2*t)*np.sin(4*t)*u(t)
plotit(ax=ax_abs, x=t, y=h, axes=True, axis=[t[0], t[-1], h.min()*1.1, h.max()*1.1], save=False, title='h(t)', xlabel='t', wait=False)
#fig.set_size_inches([6.4, 4.8])

# Polos
sp = plt.subplots(1, 1)
ax_abs = sp[0].axes[0]
plotit(ax=ax_abs, x=[], y=[], axes=True, axis=[-4, 2, -3, 3], save=False, xlabel='$\sigma$', ylabel='$j\omega$', wait=False)
ax_abs.scatter(np.array([-2, -1]), np.array([0, 0]), s=100, c='r', marker='x')
rect = patches.Rectangle((-1, -3), 3, 6, linewidth=0, edgecolor='none', facecolor='b', alpha=.2)
ax_abs.add_patch(rect)

sp = plt.subplots(1, 1)
ax_abs = sp[0].axes[0]
plotit(ax=ax_abs, x=[], y=[], axes=True, axis=[-1.5, .5, -1, 1], save=False,
       xlabel='$\sigma$', ylabel='$j\omega$', wait=False,
       xticks=[-1, .1])
ax_abs.scatter(np.array([-1, .1]), np.array([0, 0]), s=100, c='r', marker='x')
rect = patches.Rectangle((.1, -1), .4, 2, linewidth=0, edgecolor='none', facecolor='b', alpha=.2)
ax_abs.add_patch(rect)

sp = plt.subplots(1, 1)
ax_abs = sp[0].axes[0]
plotit(ax=ax_abs, x=[], y=[], axes=True, axis=[-5, 5, -5, 5], save=False,
       xlabel='$\sigma$', ylabel='$j\omega$', wait=False,
       xticks=np.linspace(-5, 5, 11))
ax_abs.scatter(np.array([-1, -1]), np.array([-4, 4]), s=100, c='r', marker='x')
rect = patches.Rectangle((-1, -5), 6, 10, linewidth=0, edgecolor='none', facecolor='b', alpha=.2)
ax_abs.add_patch(rect)

sp = plt.subplots(1, 1)
ax_abs = sp[0].axes[0]
plotit(ax=ax_abs, x=[], y=[], axes=True, axis=[-1, 1, -4.5, 4.5], save=False,
       xlabel='$\sigma$', ylabel='$j\omega$', wait=False,
       xticks=[.2], yticks=[-4, 4])
ax_abs.scatter(np.array([.2, .2]), np.array([-4, 4]), s=100, c='r', marker='x')
rect = patches.Rectangle((.2, -4.5), .8, 9, linewidth=0, edgecolor='none', facecolor='b', alpha=.2)
ax_abs.add_patch(rect)