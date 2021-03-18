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

sp = plt.subplots(3, 1)
ax_x = sp[0].axes[0]
ax_abs = sp[0].axes[1]
ax_ang = sp[0].axes[2]

A, phi, w = 1, 0, 1
plot_parts(A, phi, w, sp[0].axes, wait=True)
A, phi, w = 1, 0, 2
plot_parts(A, phi, w, sp[0].axes)
A, phi, w = 1, 0, 3
plot_parts(A, phi, w, sp[0].axes)
A, phi, w = 1, 0, 4
plot_parts(A, phi, w, sp[0].axes)
A, phi, w = 1, 0, -4
plot_parts(A, phi, w, sp[0].axes)
A, phi, w = 1, 0, 4
plot_parts(A, phi, w, sp[0].axes)
A, phi, w = 1, 0, 3
plot_parts(A, phi, w, sp[0].axes)
A, phi, w = 2, 0, 3
plot_parts(A, phi, w, sp[0].axes)
A, phi, w = 2, np.pi/4, 3
plot_parts(A, phi, w, sp[0].axes)
A, phi, w = 2, np.pi/4, 1
plot_parts(A, phi, w, sp[0].axes)
A, phi, w = 2, 0, 1
plot_parts(A, phi, w, sp[0].axes)
A, phi, w = 2, 0, .5
plot_parts(A, phi, w, sp[0].axes)
A, phi, w = 2, np.pi/4, .5
plot_parts(A, phi, w, sp[0].axes)