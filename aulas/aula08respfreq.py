from graficos import myplot as plotit
from graficos import pixaxis, piyaxis
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate
import numpy as np
import matplotlib.pyplot as plt
import time


def plot_parts(A, phi, w, ax_list, wait=False):
    ax_x = ax_list[0]
    ax_abs = ax_list[1]
    ax_ang = ax_list[2]
    axis = [-10, 10, -2.5, 2.5]
    title = "x(t) = {:.2f}".format(A) + ' exp{' + "{:.2f}".format(w) + 't'
    if phi >= 0:
        title += '+'
    else:
        title += '-'
    title += "{:.2f}".format(phi) + '}'
    t = np.arange(-15, 15, 1e-3)
    x = A * np.exp(1j * (w * t + phi))
    x_real = np.real(x)
    x_imag = np.imag(x)
    x_abs = np.abs(x)
    x_angle = np.angle(x)
    plotit(ax=ax_x, x=t, y=x_real, axes=False, axis=axis, wait=wait)
    plotit(ax=ax_x, x=t, y=x_imag, axes=False, axis=axis, color='C1', hold=True)
    ax_x.set_title(title)
    plotit(ax=ax_abs, x=t, y=x_abs, axes=False, axis=axis)
    axis[2] = -np.pi * 1.1
    axis[3] = -axis[2]
    ytick = [np.pi, np.pi/2, 0, -np.pi/2, -np.pi]
    plotit(ax=ax_ang, x=t, y=x_angle, axes=False, axis=axis, yticks=ytick)
    piyaxis(ax_ang)
    lgd = ['Real{x(t)}', 'Imag{x(t)}']
    ax_x.legend(lgd)
    ax_abs.legend(['|x(t)|'])
    ax_ang.legend(['∠x(t)'])
    plotit(ax=ax_ang, x=t, y=x_angle, axes=False, axis=axis, yticks=ytick, hold=True, save=True)

sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]

t = np.arange(-1.3, 1.3, 1e-4)
x = np.cos(10*t)
H_10 = 1/(5+1j*10)
ganho = np.abs(H_10)
fase = np.angle(H_10)
y = ganho*np.cos(10*t + fase)
axis = [t[0], t[-1], -1.1, 1.1]
plotit(ax=ax_x, x=t, y=x, axes=False, axis=axis)
plotit(ax=ax_x, x=t, y=y, axes=False, axis=axis, color='C1', hold=True, xlabel='t [s]')
plt.legend(['x(t)', 'y(t)'])

w = np.arange(-50, 50, 0.01)
env = np.divide(1, np.abs(.5*w))
env[np.where(t == 0.0)] = 1e10
plotit(ax=ax_x, x=w, y=env, axes=True, axis=axis, color='gray', linewidth=1)
H_w = np.divide(np.sin(.5*w), .5*w)
H_w[np.where(t == 0.0)] = 1
axis = [w[0], w[-1], np.min(H_w)*1.1, np.max(H_w)*1.1]
xticks = np.arange(-16*np.pi, 16*np.pi, 2*np.pi)
plotit(ax=ax_x, x=w, y=H_w, axes=True, axis=axis, hold=True, xlabel='ω [rad/s]', title='H(ω)', xticks=xticks)
pixaxis(ax_x)