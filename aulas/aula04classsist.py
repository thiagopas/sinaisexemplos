from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate
import numpy as np
import matplotlib.pyplot as plt
import time


def x_RC(t, R, C):
    return np.exp(-t/(R*C))*u(t) / (R*C)


sp = plt.subplots(1, 2)
ax_x = sp[0].axes[0]
ax_y = sp[0].axes[1]

axis = [-1, 3, -.5, 2.5]
wait = False

t = np.arange(-3, 5, .01)
x = u(t)*t/2
y = x**2
xticks = np.arange(axis[0], axis[1])
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=True)

y = 0.5 * u(t-1) * (t-1)/2
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait)

axis = [-1, 3, -1.5, 2.5]
x1 = u(t) - 2*u(t-.5) + u(t-1)
y = integrate(t, x1)
plotit(ax=ax_x, x=t, y=x1, axis=axis, xticks=xticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait)

x2 = u(t) - u(t-1.5)
y = integrate(t, x2)
plotit(ax=ax_x, x=t, y=x2, axis=axis, xticks=xticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait)

x3 = x1 + x2
y = integrate(t, x3)
plotit(ax=ax_x, x=t, y=x3, axis=axis, xticks=xticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait)

x1 = - u(t)
y1 = x1**2
plotit(ax=ax_x, x=t, y=x1, axis=axis, xticks=xticks)
plotit(ax=ax_y, x=t, y=y1, axis=axis, xticks=xticks, wait=wait)

x2 = 0.5*t*u(t)
y2 = x2**2
plotit(ax=ax_x, x=t, y=x2, axis=axis, xticks=xticks)
plotit(ax=ax_y, x=t, y=y2, axis=axis, xticks=xticks, wait=wait)

x = x1 + x2
y = x**2
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait)

x = u(t) - u(t-.5)
y = integrate(t, x)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait)

x = u(t-1) - u(t-1.5)
y = integrate(t, x)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait)

x = u(t) - u(t-.5)
y = x * np.cos(t)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait)

x = u(t-1) - u(t-1.5)
y = x * np.cos(t)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait)

yticks = np.arange(-1, 3, .5)
x = 1.3*u(t) - .3*u(t-1.5)
y = .5*x
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, yticks=yticks)

x = (t/1.5)*(u(t)-u(t-1.5)) + u(t-1.5)
y = .5*x
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, yticks=yticks)

x = 1.3*u(t) - .3*u(t-1.5)
y = x**2
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, yticks=yticks)

x = (t/1.5)*(u(t)-u(t-1.5)) + u(t-1.5)
y = x**2
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, yticks=yticks)

x = 1.3*u(t) - .3*u(t-1.5)
y = integrate(t, x)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, yticks=yticks)

x = (t/1.5)*(u(t)-u(t-1.5)) + u(t-1.5)
y = integrate(t, x)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, yticks=yticks)

x = 1.3*u(t) - .3*u(t-1.5)
y = 1.3*u(t-.5) - .3*u(t-2)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, yticks=yticks)

x = (t/1.5)*(u(t)-u(t-1.5)) + u(t-1.5)
y = ((t-.5)/1.5)*(u(t-.5)-u(t-2)) + u(t-2)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, yticks=yticks)

R = 1
C = 0.5
x = 1.3*u(t) - .3*u(t-1.5)
h = x_RC(t, R, C)
y = conv(x, h, t)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, yticks=yticks)

x = (t/1.5)*(u(t)-u(t-1.5)) + u(t-1.5)
y = conv(x, h, t)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, yticks=yticks)

x = 1.3*u(t) - .3*u(t-1.5)
y = movavgpast(t, x, 1.)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=False, yticks=yticks)

x = (t/1.5)*(u(t)-u(t-1.5)) + u(t-1.5)
y = movavgpast(t, x, 1.)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=False, yticks=yticks)

x = 1.3*u(t) - .3*u(t-1.5)
y = movavgc(t, x, 1.)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=False, yticks=yticks)

x = (t/1.5)*(u(t)-u(t-1.5)) + u(t-1.5)
y = movavgc(t, x, 1.)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=False, yticks=yticks)

x = u(t)
y = movavgpast(t, x, 1.)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=False, yticks=yticks)

y = integrate(t, x)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=False, yticks=yticks)

x = u(t)
y = np.zeros_like(x)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=False, yticks=yticks)

x = .4*(u(t)-u(t-2))*(-np.cos(t*np.pi)+1)
y = derivate(t, x)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=False, yticks=yticks)

axis = [-1, 8, -2, 2]
t = np.arange(-3, 14, .01)
xticks = np.arange(axis[0], axis[1])
yticks = np.arange(np.ceil(axis[2]), axis[3], .5)

def pulso(t):
    return u(t)-u(t-1)

def ht(t):
    return u(t)-u(t-3)

x = pulso(t)
y = ht(t)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=False, yticks=yticks)


axis = [-1, 8, -1.2, 1.2]
x = pulso(t)
y = ht(t)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=True, yticks=yticks)

x = .5*pulso(t-1)
y = .5*ht(t-1)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=True, yticks=yticks)

x = - .5*pulso(t-3)
y = - .5*ht(t-3)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=True, yticks=yticks)

x = - pulso(t-4)
y = - ht(t-4)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=True, yticks=yticks)

axis = [-1, 8, -2, 2]
x = pulso(t) + .5*pulso(t-1) - .5*pulso(t-3) - pulso(t-4)
y = ht(t) + .5*ht(t-1) - .5*ht(t-3) - ht(t-4)
plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y, x=t, y=y, axis=axis, xticks=xticks, wait=wait, save=True, yticks=yticks)

plotit(ax=ax_x, x=t, y=x, axis=axis, xticks=xticks, yticks=yticks)
plotit(ax=ax_y,axis=axis, xticks=xticks, wait=wait, save=True, yticks=yticks)