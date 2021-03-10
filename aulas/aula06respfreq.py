from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate
import numpy as np
import matplotlib.pyplot as plt
import time


def x_RC(t, R, C):
    return np.exp(-t/(R*C))*u(t) / (R*C)

def w_RL(w, R, L):
    return (1/L) * 1/(R/L + 1j*w)

def piyaxis(ax):
    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(["{:.1f}".format(y / np.pi) + 'π' for y in ticks_loc])

def pixaxis(ax):
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels(["{:.0f}".format(x / np.pi) + 'π' for x in ticks_loc])

sp = plt.subplots(3, 1)
ax_x = sp[0].axes[0]
ax_abs = sp[0].axes[1]
ax_ang = sp[0].axes[2]

R = 2
L = 1e-2
t = np.arange(-16e-3, 2*16e-3, 1e-4)

w = np.arange(0, 200*2*np.pi)
xticks_w = np.arange(0, w[-1], 60*np.pi)
resp = w_RL(w, 2, 1e-2)

f_array = np.arange(10, 200, 10)

plotit(ax=ax_x, wait=True)

for f in f_array:
    w0 = 2*np.pi*f
    x = 220*np.cos(w0*t)
    resp0 = w_RL(w0, R, L)
    y = 220*np.abs(resp0)*np.cos(w0*t + np.angle(resp0))
    str_x = 'x(t)=220cos(2π'+"{:.0f}".format(f)+')'
    str_y = 'y(t)=' + "{:.1f}".format(np.abs(220*resp0)) + 'cos(2π'+"{:.0f}".format(f) + \
            "{:.2f}".format(np.angle(resp0)/np.pi) + 'π)'

    wait = False

    axis_abs = [w[0], w[-1], 0, np.max(np.abs(resp))*1.1]
    axis_ang = [w[0], w[-1], np.min(np.angle(resp))*1.1, 0]
    axis_t = [t[0], t[-1], -220*1.1, 220*1.1]

    plotit(ax=ax_x, x=t, y=x, axis=axis_t, axes=False)
    plotit(ax=ax_x, x=t, y=y, axis=axis_t, color='C1', hold=True, axes=False)
    ax_x.legend([str_x, str_y])
    plotit(ax=ax_abs, x=w, y=np.abs(resp), axis=axis_abs, axes=False, xticks=xticks_w)
    ax_abs.plot([w0], [np.abs(resp0)], 'o', color='red', markersize=10)
    pixaxis(ax_abs)
    str_line = '|H(ω)|'
    str_dot = '|H(2π' + "{:.0f}".format(f) + ')|=' + "{:.2f}".format(np.abs(resp0))
    ax_abs.legend([str_line, str_dot])
    plotit(ax=ax_ang, x=w, y=np.angle(resp), axis=axis_ang, axes=False, xticks=xticks_w)
    ax_ang.plot([w0], [np.angle(resp0)], 'o', color='red', markersize=10)
    piyaxis(ax_ang)
    pixaxis(ax_ang)
    str_line = '∠H(ω)'
    str_dot = '∠H(2π' + "{:.0f}".format(f) + ')=' + "{:.2f}".format(np.angle(resp0)/np.pi) + 'π'
    ax_ang.legend([str_line, str_dot])
    plotit(ax=ax_abs, hold=True, axes=False, save=True)


#plotit(ax=ax_x, x=w, y=np.abs(resp), axis=axis_abs)#, xticks=xticks)
#plotit(ax=ax_x, x=w, y=np.angle(resp), wait=True, axis=axis_ang)#, wait=True, xticks=xticks, wait=True)
