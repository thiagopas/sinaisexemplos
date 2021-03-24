from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate
import numpy as np
import matplotlib.pyplot as plt
import time


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

def pixaxis(ax):
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels(["{:.0f}".format(x / np.pi) + 'π' for x in ticks_loc])

sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]

R = 2
L = 1e-2
t = np.arange(-16e-3, 2*16e-3, 1e-4)

w = np.arange(0, 200*2*np.pi)
xticks_w = np.arange(0, w[-1], 60*np.pi)
resp = w_RL(w, 2, 1e-2)

f_array = np.arange(10, 200, 10)

w0 = 2 * np.pi * np.array([60, 120, 150, 250, 270])/2
resp0 = w_RL(w0, R, L)
wait = False

axis_abs = [w[0], w[-1], 0, np.max(np.abs(resp)) * 1.1]
axis_ang = [w[0], w[-1], np.min(np.angle(resp)) * 1.1, 0]
axis_t = [t[0], t[-1], -220 * 1.1, 220 * 1.1]

plotit(ax=ax_abs, x=w, y=np.abs(resp), axis=axis_abs, axes=False, xticks=xticks_w)
str_line = '|H(ω)|'
ax_abs.legend([str_line])
pixaxis(ax_abs)
ax_abs.plot(w0, np.abs(resp0), 'o', color='red', markersize=7)
plotit(ax=ax_ang, x=w, y=np.angle(resp), axis=axis_ang, axes=False, xticks=xticks_w)
ax_ang.legend([str_line])
ax_ang.plot(w0, np.angle(resp0), 'o', color='red', markersize=7)
piyaxis(ax_ang)
pixaxis(ax_ang)
str_line = '∠H(ω)'
plt.legend([str_line])
plotit(ax=ax_abs, hold=True, axes=False, save=False)

sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]
w = np.arange(-120*2*np.pi, 110*2*np.pi)
resp = w_RL(w, 2, 1e-2)
axis_abs = [w[0], w[-1], 0, np.max(np.abs(resp)) * 1.1]
axis_ang = [w[0], w[-1], np.min(np.angle(resp)) * 1.1, np.max(np.angle(resp)) * 1.1]
xticks_w = np.arange(w[0], w[-1], 60*np.pi)
w0 = 2 * np.pi * np.arange(-360, 360, 30)/2
resp0 = w_RL(w0, R, L)
plotit(ax=ax_abs, x=w, y=np.abs(resp), axis=axis_abs, axes=False, xticks=xticks_w)
str_line = '|H(ω)|'
ax_abs.legend([str_line])
pixaxis(ax_abs)
ax_abs.plot(w0, np.abs(resp0), 'o', color='red', markersize=7)
plotit(ax=ax_ang, x=w, y=np.angle(resp), axis=axis_ang, axes=False, xticks=xticks_w)
ax_ang.legend([str_line])
ax_ang.plot(w0, np.angle(resp0), 'o', color='red', markersize=7)
piyaxis(ax_ang)
pixaxis(ax_ang)
str_line = '∠H(ω)'
plt.legend([str_line])
plotit(ax=ax_abs, hold=True, axes=False, save=False)


#plotit(ax=ax_x, x=w, y=np.abs(resp), axis=axis_abs)#, xticks=xticks)
#plotit(ax=ax_x, x=w, y=np.angle(resp), wait=True, axis=axis_ang)#, wait=True, xticks=xticks, wait=True)

t = np.arange(-.15, .15, 1e-4)
def dn(n):
    n = float(n)
    if n == 0:
        return .5
    else:
        return np.sin(n*.5*np.pi)/(n*np.pi)


def dn_n(nvec):
    d_values = np.zeros_like(nvec, dtype=float)
    for i in range(len(nvec)):
        d_values[i] = dn(nvec[i])
    return d_values

def synth(t, n_final, w0):
    x = np.zeros_like(t)
    for n in np.arange(-n_final, n_final+1):
        x += np.real(dn(n)*np.exp(1j*n*w0*t))
    return x

def synth_filtered(t, n_final, w0):
    x = np.zeros_like(t)
    for n in np.arange(-n_final, n_final+1):
        x += np.real(w_RC(n*w0, 1, 1e-2)*dn(n)*np.exp(1j*n*w0*t))
    return x

def square(t, t0, t_init, min, max, duty):
    t_per = np.mod(t-t_init, t0)/t0
    x = np.zeros_like(t)
    x[np.where(np.abs(t_per) <= duty)] = max
    x[np.where(np.abs(t_per) > duty)] = min
    return x

sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
x = square(t, .1, -.025, 0, 1, .5)
y = synth_filtered(t, 1, 20*np.pi)
axis = [t[0], t[-1], -.1, 1.1]
plotit(ax=ax_x, x=t, y=x, axes=True, axis=axis, linewidth=2, wait=False, save=False)
sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]

t_ = np.arange(-.1, .3, 1e-5)
h = x_RC(t_, 1, 10e-3)
axis = [t_[0], t_[-1], -10, h.max()*1.1]
plotit(ax=ax_x, x=t_, y=h, axes=True, axis=axis, linewidth=2, wait=False, save=False)

w0 = 20*np.pi
n = np.arange(-30, 21)
#d_values = w_RC(n*w0, 1, 10e-3)
d_values = np.zeros_like(n, dtype=float)
for i in range(len(n)):
    d_values[i] = dn(n[i])
sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]
ax_abs.stem(n*w0, np.abs(d_values))
ax_abs.grid()
ax_abs.set_xticks(np.arange(-300*np.pi, 360*np.pi, 60*np.pi))
pixaxis(ax_abs)
ax_abs.axis([-200*np.pi, 200*np.pi, -.05, d_values.max()*1.1])
ax_abs.legend(['$|D_n|$'])
angles = np.angle(d_values)
angles[np.where(abs(d_values) < 1e-10)] = 0
angles = angles * np.sign(n)
ax_ang.stem(n*w0, angles)
ax_ang.grid()
ax_ang.set_xticks(np.arange(-300*np.pi, 360*np.pi, 60*np.pi))
pixaxis(ax_ang)
ax_ang.axis([-200*np.pi, 200*np.pi, -4, 4])
ax_ang.set_yticks(np.arange(-np.pi, 1.5*np.pi, .5*np.pi))
piyaxis(ax_ang)
ax_ang.legend(['$∠D_n$'])

sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]
w = np.arange(-300*np.pi, 360*np.pi)
resp = w_RC(w, 1, 10e-3)
axis_abs = [-200*np.pi, 200*np.pi, 0, np.max(np.abs(resp)) * 1.1]
axis_ang = [-200*np.pi, 200*np.pi, np.min(np.angle(resp)) * 1.1, np.max(np.angle(resp)) * 1.1]
xticks_w = np.arange(w[0], w[-1], 60*np.pi)
w0 = 2 * np.pi * np.arange(-360, 360, 30)/2
resp0 = w_RC(w0, 1, 10e-3)
plotit(ax=ax_abs, x=w, y=np.abs(resp), axis=axis_abs, axes=False, xticks=xticks_w)
str_line = '|H(ω)|'
ax_abs.legend([str_line])
pixaxis(ax_abs)
#ax_abs.plot(w0, np.abs(resp0), 'o', color='red', markersize=7)
ax_abs.axis(axis_abs)
plotit(ax=ax_ang, x=w, y=np.angle(resp), axis=axis_ang, axes=False, xticks=xticks_w)
ax_ang.legend([str_line])
#ax_ang.plot(w0, np.angle(resp0), 'o', color='red', markersize=7)
piyaxis(ax_ang)
pixaxis(ax_ang)
str_line = '∠H(ω)'
plt.legend([str_line])
ax_ang.axis(axis_ang)

# Resultado da multiplicacao
w0 = 20*np.pi
n = np.arange(-30, 21)
d_values = np.zeros_like(n, dtype=complex)
for i in range(len(n)):
    d_values[i] = dn(n[i]) * w_RC(n[i] * w0, 1, 10e-3)
sp = plt.subplots(2, 1)
ax_abs = sp[0].axes[0]
ax_ang = sp[0].axes[1]
ax_abs.stem(n*w0, np.abs(d_values))
ax_abs.grid()
ax_abs.set_xticks(np.arange(-300*np.pi, 360*np.pi, 60*np.pi))
pixaxis(ax_abs)
ax_abs.axis([-200*np.pi, 200*np.pi, -.05, np.abs(d_values).max()*1.1])
ax_abs.legend(['$|H(\omega_0 n)| |D_n|$'])
angles = np.angle(d_values)
angles[np.where(abs(d_values) < 1e-10)] = 0
#angles = angles * np.sign(n)
ax_ang.stem(n*w0, angles)
ax_ang.grid()
ax_ang.set_xticks(np.arange(-300*np.pi, 360*np.pi, 60*np.pi))
pixaxis(ax_ang)
ax_ang.axis([-200*np.pi, 200*np.pi, -4, 4])
ax_ang.set_yticks(np.arange(-np.pi, 1.5*np.pi, .5*np.pi))
piyaxis(ax_ang)
ax_ang.legend(['$∠H(\omega_0 n)+∠D_n$'])

# Saída (domínio do tempo)
sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
y = synth_filtered(t, 100, 20*np.pi)
axis = [t[0], t[-1], -.1, 1.1]
plotit(ax=ax_x, x=t, y=y, axes=True, axis=axis, linewidth=2, wait=False, save=False)

sp = plt.subplots(2, 1)
ax_x = sp[0].axes[0]
ax_y = sp[0].axes[1]
plotit(ax=ax_x, x=t, y=y, axes=True, axis=axis, linewidth=2, wait=True, save=False)
for base in range(12):
    n_limit = 2 ** base
    x = synth(t, n_limit, 20 * np.pi)
    y = synth_filtered(t, n_limit, 20 * np.pi)
    plotit(ax=ax_x, x=t, y=x, axes=True, axis=axis, linewidth=2, wait=False, save=False, title='N='+str(n_limit))
    plotit(ax=ax_y, x=t, y=y, axes=True, axis=axis, linewidth=2, wait=False, save=True, color='C1')

if False:
    file = open('../temp/coefs.html', 'w')
    file.write('<html><head>')
    file.write('<style>table, td, th {  border: 1px solid black;} table {border-collapse: collapse;} </style>')
    file.write('</head>')
    file.write('<body><table border=1>\n')
    n_max = 10
    line_1 = ''
    line_2 = ''
    for n in np.arange(-n_max, n_max+1):
        d = dn(n)
        if np.abs(d) < 1e-6:
            d = int(0)
        line_1 += '<td align=center>D<sub>' + str(n) + '</sub></td>'
        line_2 += '<td>&nbsp;' + "{:.3f}".format(d) + '&nbsp;</td>'
    file.write('<tr>' + line_1 + '</tr>\n')
    file.write('<tr>' + line_2 + '</tr>\n')
    file.write('</table></body></html>')
    file.close()



