from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate
import numpy as np
import matplotlib.pyplot as plt
import time


def x_RC(t, R, C):
    return np.exp(-t/(R*C))*u(t) / (R*C)

def w_RC(w, R, C):
    return (1 / (1 + 1j*w*(R*C)))

def piyaxis(ax):
    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(["{:.2f}".format(y / np.pi) + 'π' for y in ticks_loc])

def pixaxis(ax):
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels(["{:.0f}".format(x / np.pi) + 'π' for x in ticks_loc])

R = 1.
C = 10.0e-3

sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]
w = np.arange(0, 500)
resp = w_RC(w, R, C)
plotit(ax=ax_abs, x=w, y=np.abs(resp), ylabel='$|H(\omega)|$', axis=[w[0], w[-1], -.1, 1.1], save=False)
ytick = np.array([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
plotit(ax=ax_ang, x=w, y=np.angle(resp), axes=True, xlabel='$\omega$ [rad/s]', ylabel='$∠H(\omega)$', yticks=ytick, axis=[w[0], w[-1], -np.pi/2, np.pi/2], save=False)
piyaxis(ax_ang)
ax_ang.set_xlabel('$\omega$ [rad/s]')

sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]
w = np.arange(0, 10000)
resp = w_RC(w, R, C)
plotit(ax=ax_abs, x=w, y=np.abs(resp), ylabel='$|H(\omega)|$', axis=[w[0], w[-1], -.1, 1.1], wait=True, save=False)
ytick = np.array([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
plotit(ax=ax_ang, x=w, y=np.angle(resp), axes=True, xlabel='$\omega$ [rad/s]', ylabel='$∠H(\omega)$', yticks=ytick, axis=[w[0], w[-1], -np.pi/2, np.pi/2], save=False)
piyaxis(ax_ang)


sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]
w = np.arange(0, 10000)
resp = w_RC(w, R, C)
plotit(ax=ax_abs, x=w, y=np.abs(resp), ylabel='$|H(\omega)|$', axis=[w[0], w[-1], -.01, .05], wait=True, save=False)
ytick = np.array([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
plotit(ax=ax_ang, x=w, y=np.angle(resp), axes=True, xlabel='$\omega$ [rad/s]', ylabel='$∠H(\omega)$', yticks=ytick, axis=[w[0], w[-1], -np.pi/2, np.pi/2], save=False)
piyaxis(ax_ang)

sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]
w = np.arange(0, 10000)
resp = w_RC(w, R, C)
plotit(ax=ax_abs, x=w, y=np.abs(resp), ylabel='$|H(\omega)|$', axis=[w[0], w[-1], -.01, .05], wait=True, save=False)
ytick = np.array([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
plotit(ax=ax_ang, x=w, y=np.angle(resp), axes=True, xlabel='$\omega$ [rad/s]', ylabel='$∠H(\omega)$', yticks=ytick, axis=[w[0], w[-1], -np.pi/2, np.pi/2], save=False)
piyaxis(ax_ang)

sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]
w = np.arange(.1, 10001)
resp = w_RC(w, R, C)
ax_abs.semilogy(w, np.abs(resp))
ax_abs.grid()
ytick = np.array([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
plotit(ax=ax_ang, x=w, y=np.angle(resp), axes=True, xlabel='$\omega$ [rad/s]', ylabel='$∠H(\omega)$', yticks=ytick, axis=[w[0], w[-1], -np.pi/2, np.pi/2], save=False)
piyaxis(ax_ang)
ax_abs.set_ylabel('$|H(\omega)|$')
ax_ang.set_ylabel('$∠H(\omega)$')
ax_ang.set_xlabel('$\omega$ [rad/s]')

sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]
w = np.arange(.1, 10001)
resp = w_RC(w, R, C)
ax_abs.loglog(w, np.abs(resp))
ax_abs.grid()
ytick = np.array([-np.pi/2, -np.pi/4, 0])
ax_ang.set_yticks(ytick)
ax_ang.grid()
ax_ang.semilogx(w, np.angle(resp))
#plotit(ax=ax_ang, x=w, y=np.angle(resp), axes=True, xlabel='$\omega$ [rad/s]', ylabel='$∠H(\omega)$', yticks=ytick, axis=[w[0], w[-1], -np.pi/2, np.pi/2])
piyaxis(ax_ang)
ax_abs.set_ylabel('$|H(\omega)|$')
ax_ang.set_ylabel('$∠H(\omega)$')
ax_ang.set_xlabel('$\omega$ [rad/s]')