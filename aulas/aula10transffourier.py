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

sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
t = np.arange(-.1, .1, 1e-5)
axis = [-.02, .08, -.1, 1.1]
# Onda quadrada
x = square(t, .005, -.0025/2, 0., 1., .5)
plotit(ax=ax_x, x=t, y=x, axis=axis, axes=True, title='x(t)', save=False)
h = x_RC(t, 1, 10e-3)
y = conv(x, h, t)
ax_x = plt.subplots(1, 1)[0].axes[0]
plotit(ax=ax_x, x=t, y=y, axis=axis, axes=True, title='y(t)', save=False)

# Pulso
x = u(t+.0025/2) - u(t-.0025/2)
ax_x = plt.subplots(1, 1)[0].axes[0]
plotit(ax=ax_x, x=t, y=x, axis=axis, axes=True, title='x(t)', save=False)
h = x_RC(t, 1, 10e-3)
y = conv(x, h, t)
ax_x = plt.subplots(1, 1)[0].axes[0]
plotit(ax=ax_x, x=t, y=y, axis=axis, axes=True, title='y(t)', save=False)

# Sequencia com período crescente
sp = plt.subplots(3, 1)
ax_x = sp[0].axes[0]
ax_abs = sp[0].axes[1]
ax_ang = sp[0].axes[2]
w0_0 = 2*np.pi/.005
n = np.arange(-50, 50)
for i in np.arange(0, 10, 1):
    t0 = .005*(i+1)
    w0 = 2*np.pi / t0
    d = dn_n(n, t0)
    x = square(t, t0, -.005/4, 0., 1., .5/(i+1))
    ax_abs.cla()
    ax_abs.stem(w0*n, np.abs(d))
    ax_abs.axis([-3*w0_0, 3*w0_0, -.05, .55])
    ax_abs.legend(['$|D_n|$'])
    ax_abs.grid()
    xtick = np.arange(-3, 4) * w0_0
    ax_abs.set_xticks(xtick)
    pixaxis(ax_abs)
    ax_ang.cla()
    ax_ang.stem(w0 * n, np.angle(d)*np.sign(n))
    ax_ang.axis([-3 * w0_0, 3 * w0_0, -np.pi*1.1, np.pi*1.1])
    plt.legend(['$∠D_n$'])
    ax_ang.set_xticks(xtick)
    pixaxis(ax_ang)
    plt.grid()
    plotit(ax=ax_x, x=t, y=x, axis=[-.05, .05, -.1, 1.1], axes=False, save=False, legend=['x(t)'])

# Aperiódico com espectro contínuo
width = .0025
x = window(t, -width/2, width/2)
plotit(ax=ax_x, x=t, y=x, axis=[-.05, .05, -.1, 1.1], axes=False, save=False, legend=['x(t)'])
w = np.arange(-3*w0_0, 3*w0_0)
def thesinc(w, width):
    return width*np.sin(w*width/2) / (w*width/2)
x_w = thesinc(w, width)
xtick = np.arange(-3, 4) * w0_0
plotit(ax=ax_abs, x=w, y=np.abs(x_w), axis=[w[0], w[-1], 0., np.max(np.abs(x_w))*1.1], axes=False, save=False, legend=['$|X(\omega)$|'])
ax_abs.set_xticks(xtick)
pixaxis(ax_abs)
plotit(ax=ax_ang, x=w, y=np.angle(x_w)*np.sign(w), axis=[w[0], w[-1], -np.pi*1.1, np.pi*1.1], axes=False, save=True, legend=['$∠X(\omega)$'])
ax_ang.set_xticks(xtick)
pixaxis(ax_ang)

# Aperiódico com espectro contínuo
# (exibição de espectro mais completo)
w = np.arange(-3*w0_0*3, 3*w0_0*3)
def thesinc(w, width):
    return width*np.sin(w*width/2) / (w*width/2)
x_w = thesinc(w, width)
xtick = np.arange(-3*3, 4*3, 3) * w0_0
plotit(ax=ax_abs, x=w, y=np.abs(x_w), axis=[w[0], w[-1], 0., np.max(np.abs(x_w))*1.1], axes=False, save=False, legend=['$|X(\omega)$|'])
ax_abs.set_xticks(xtick)
pixaxis(ax_abs)
plotit(ax=ax_ang, x=w, y=np.angle(x_w)*np.sign(w), axis=[w[0], w[-1], -np.pi*1.1, np.pi*1.1], axes=False, save=False, legend=['$∠X(\omega)$'])
ax_ang.set_xticks(xtick)
pixaxis(ax_ang)

# Sinc(x)
sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
x = np.arange(-6*np.pi, 6*np.pi, 1e-3)
plotit(ax=ax_x, x=x, y=sinc(x), axis=[x[0], x[-1], -.3, 1.1], axes=False, save=True, legend=['sinc(x)'], xlabel='x')

# sinc na frequencia
sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]
w = np.arange(-9*np.pi, 9*np.pi, 1e-3)
x = sinc(w/2)
xtick = np.arange(-8*np.pi, 9*np.pi, np.pi*2)
plotit(ax=ax_abs, x=w, y=np.abs(x), axis=[w[0], w[-1], -.1, 1.1], axes=False, save=True, legend=['$|\\tau sinc(\omega\\tau/2)|$'], xlabel='$\omega$', xticks=xtick)
ax_abs.set_xticklabels(['$-8\pi/\\tau$', '$-6\pi/\\tau$', '$-4\pi/\\tau$', '$-2\pi/\\tau$', '0', '$2\pi/\\tau$', '$4\pi/\\tau$', '$6\pi/\\tau$', '$8\pi/\\tau$'])
plotit(ax=ax_ang, x=w, y=np.angle(x)*np.sign(w), axis=[w[0], w[-1], -np.pi*1.1, np.pi*1.1], axes=False, save=True, legend=['$∠\\tau sinc(\omega\\tau/2)$'], xlabel='$\omega$', xticks=xtick)
ax_ang.set_xticklabels(['$-8\pi/\\tau$', '$-6\pi/\\tau$', '$-4\pi/\\tau$', '$-2\pi/\\tau$', '0', '$2\pi/\\tau$', '$4\pi/\\tau$', '$6\pi/\\tau$', '$8\pi/\\tau$'])
ax_ang.set_yticks([-np.pi, 0, np.pi])
piyaxis(ax_ang)
tauyaxis(ax_abs)

# espectro do sinal exemplo
sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]
w = np.arange(-9*400*np.pi, 9*400*np.pi)
tau = 0.0025
x = tau * sinc(w*tau/2)
xtick = np.arange(-8*400*np.pi, 9*400*np.pi, np.pi*2*400*2)
plotit(ax=ax_abs, x=w, y=np.abs(x), axis=[w[0], w[-1], -.1*tau, 1.1*tau], axes=False, save=False, legend=['$|X(\omega)|$'], xlabel='$\omega$', xticks=xtick, wait=True)
ax_abs.set_xticklabels(['$-3200\pi$', '$-1600\pi$', '0', '$1600\pi$', '$3200\pi$'])
plotit(ax=ax_ang, x=w, y=np.angle(x)*np.sign(w), axis=[w[0], w[-1], -np.pi*1.1, np.pi*1.1], axes=False, save=False, legend=['$∠X(\omega)$'], xlabel='$\omega$', xticks=xtick)
ax_ang.set_xticklabels(['$-3200\pi$', '$-1600\pi$', '0', '$1600\pi$', '$3200\pi$'])
ax_ang.set_yticks([-np.pi, 0, np.pi])
piyaxis(ax_ang)
plotit(ax=ax_ang, x=[], y=[], hold=True, save=True)

# resposta em frequÊncia
w = np.arange(-9*400*np.pi, 9*400*np.pi)
h= w_RC(w, 1, 10e-3)
xtick = np.arange(-8*400*np.pi, 9*400*np.pi, np.pi*2*400*2)
plotit(ax=ax_abs, x=w, y=np.abs(h), axis=[w[0], w[-1], -.1, 1.1], axes=False, save=False, legend=['$|H(\omega)|$'], xlabel='$\omega$', xticks=xtick, wait=True)
ax_abs.set_xticklabels(['$-3200\pi$', '$-1600\pi$', '0', '$1600\pi$', '$3200\pi$'])
plotit(ax=ax_ang, x=w, y=np.angle(h), axis=[w[0], w[-1], -np.pi*1.1, np.pi*1.1], axes=False, save=False, legend=['$∠H(\omega)$'], xlabel='$\omega$', xticks=xtick)
ax_ang.set_xticklabels(['$-3200\pi$', '$-1600\pi$', '0', '$1600\pi$', '$3200\pi$'])
ax_ang.set_yticks([-np.pi, 0, np.pi])
piyaxis(ax_ang)
plotit(ax=ax_ang, x=[], y=[], hold=True, save=True)

# Saída
y = x*h
xtick = np.arange(-8*400*np.pi, 9*400*np.pi, np.pi*2*400*2)
plotit(ax=ax_abs, x=w, y=np.abs(y), axis=[w[0], w[-1], -.1*np.max(np.abs(y)), np.max(np.abs(y))*1.1], axes=False, save=False, legend=['$|Y(\omega)|$'], xlabel='$\omega$', xticks=xtick, wait=True)
ax_abs.set_xticklabels(['$-3200\pi$', '$-1600\pi$', '0', '$1600\pi$', '$3200\pi$'])
plotit(ax=ax_ang, x=w, y=np.angle(y), axis=[w[0], w[-1], -np.pi*1.1, np.pi*1.1], axes=False, save=False, legend=['$∠Y(\omega)$'], xlabel='$\omega$', xticks=xtick)
ax_ang.set_xticklabels(['$-3200\pi$', '$-1600\pi$', '0', '$1600\pi$', '$3200\pi$'])
ax_ang.set_yticks([-np.pi, 0, np.pi])
piyaxis(ax_ang)
plotit(ax=ax_ang, x=[], y=[], hold=True, save=True)

