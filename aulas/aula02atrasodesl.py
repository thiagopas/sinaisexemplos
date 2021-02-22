from graficos import myplot as plotit
from funcoes import u, window
import numpy as np
import matplotlib.pyplot as plt


def xfunc(t):
    x = (np.exp(-2 * t) - np.exp(-15 * t)) * u(t)
    x = 2 * x / x.max()
    return x


sp = plt.subplots(3, 2)
ax_x = sp[0].axes[0]
ax_xi = sp[0].axes[3]
ax_xid = sp[0].axes[5]
ax_x_ = sp[0].axes[1]
ax_xd = sp[0].axes[2]
ax_xdi = sp[0].axes[4]
axis = [-6, 6, -.5, 2.5]
xticks = np.arange(-6, 6)

plotit(ax=ax_x, axis=axis, xticks=xticks, title='Deslocamento e reversão')
plotit(ax=ax_x_, axis=axis, xticks=xticks, title='Reversão e deslocamento')
plotit(ax=ax_xd, axis=axis, xticks=xticks)
plotit(ax=ax_xdi, axis=axis, xticks=xticks)
plotit(ax=ax_xi, axis=axis, xticks=xticks)
plotit(ax=ax_xid, axis=axis, xticks=xticks)
t = np.arange(-6, 6, .01)

# Deslocamento seguido de reversão
x = xfunc(t)
plotit(text='x(t)', ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, wait=True, title='Deslocamento e reversão')
plotit(text='v(t) = x(t-1)', ax=ax_xd, axis=axis, xticks=xticks, wait=True)
xd = xfunc(t - 1)
plotit(text='v(t) = x(t-1)', ax=ax_xd, x=t, y=xd, axis=axis, xticks=xticks, wait=True)
xd = xfunc(t - 2)
plotit(text='v(t) = x(t-2)', ax=ax_xd, x=t, y=xd, axis=axis, xticks=xticks, wait=True)
xd = xfunc(t - 3)
plotit(text='v(t) = x(t-3)', ax=ax_xd, x=t, y=xd, axis=axis, xticks=xticks, wait=True)
xd = xfunc(t + 3)
plotit(text='v(t) = x(t+3)', ax=ax_xd, x=t, y=xd, axis=axis, xticks=xticks, wait=True)
xd = xfunc(t - 3)
plotit(text='v(t) = x(t-3)', ax=ax_xd, x=t, y=xd, axis=axis, xticks=xticks, wait=True)
xdi = xfunc(-t - 3)
plotit(text='w(t) = v(-t)', ax=ax_xdi, axis=axis, xticks=xticks, wait=True)
plotit(text='w(t) = v(-t)', ax=ax_xdi, x=t, y=xdi, axis=axis, xticks=xticks, wait=True)

# Reversão seguida de deslocamento
plotit(text='x(t)', ax=ax_x_, x=t, y=x, axis=axis, xticks=xticks, wait=True, title='Reversão e deslocamento')
xi = xfunc(-t)
plotit(text='y(t) = x(-t)', ax=ax_xi, axis=axis, xticks=xticks, wait=True)
plotit(text='y(t) = x(-t)', ax=ax_xi, x=t, y=xi, axis=axis, xticks=xticks, wait=True)
xid = xfunc(-t + 3)
plotit(text='z(t) = y(t - 3)', ax=ax_xid, axis=axis, xticks=xticks, wait=True)
plotit(text='z(t) = y(t - 3)', ax=ax_xid, x=t, y=xid, axis=axis, xticks=xticks, wait=True)

# Notação usada na convolução
plotit(ax=ax_x, axis=axis, xticks=xticks, wait=True)
plotit(ax=ax_xd, axis=axis, xticks=xticks)
plotit(ax=ax_xdi, axis=axis, xticks=xticks, wait=True)
plotit(text='z(t) = y(t - 3) = x(-t + 3)', ax=ax_xid, x=t, y=xid, axis=axis, xticks=xticks, wait=True)
plotit(text='x(-t + 3)', ax=ax_xid, x=t, y=xid, axis=axis, xticks=xticks, wait=True)
plotit(text='x(-t + 3) = x(3 - t)', ax=ax_xid, x=t, y=xid, axis=axis, xticks=xticks, wait=True)
plotit(text='x(3 - t)', ax=ax_xid, x=t, y=xid, axis=axis, xticks=xticks, wait=True)
xid = xfunc(0 - t)
plotit(text='x(0 - t)', ax=ax_xid, x=t, y=xid, axis=axis, xticks=xticks, wait=True)
xid = xfunc(1 - t)
plotit(text='x(1 - t)', ax=ax_xid, x=t, y=xid, axis=axis, xticks=xticks, wait=True)
xid = xfunc(2 - t)
plotit(text='x(2 - t)', ax=ax_xid, x=t, y=xid, axis=axis, xticks=xticks, wait=True)
xid = xfunc(3 - t)
plotit(text='x(3 - t)', ax=ax_xid, x=t, y=xid, axis=axis, xticks=xticks, wait=True)
xid = xfunc(4 - t)
plotit(text='x(4 - t)', ax=ax_xid, x=t, y=xid, axis=axis, xticks=xticks, wait=True)
