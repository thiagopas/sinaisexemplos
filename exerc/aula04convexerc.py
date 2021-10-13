from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate, exp_u, sinc
import numpy as np
import matplotlib.pyplot as plt


t = np.arange(-1, 5, .01)
h = u(t) - u(t-1)
x = np.cos(3*np.pi*t)*u(t)
ya = conv(x, h, t)
x = np.cos(2*np.pi*t)*u(t)
yb = conv(x, h, t)

plt.subplot(3,1,1)
plt.plot(t, h)
plt.subplot(3,1,2)
plt.plot(t, x)
plt.subplot(3,1,3)
plt.plot(t, ya)

plt.figure()
plt.subplot(2,1,1)
plt.plot(t, ya)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('Questão (1a)')
plt.legend(['y(t)'])
plt.subplot(2,1,2)
plt.plot(t, yb)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('Questão (1b)')
plt.legend(['y(t)'])


# Quiz
def x_t(t):
    return u(t) - u(t-1)

plt.figure()
plt.plot(t, x_t(t), linewidth=3)
plt.grid()
plt.legend(['x(t)'])
plt.gcf().set_size_inches(6.4, 2.33)

plt.figure()
y = x_t(t)
plt.plot(t, y, linewidth=3)
plt.grid()
plt.legend(['y(t)'])
plt.gcf().set_size_inches(6.4, 2.33)

plt.figure()
y = x_t(t-1)
plt.plot(t, y, linewidth=3)
plt.grid()
plt.legend(['y(t)'])
plt.gcf().set_size_inches(6.4, 2.33)

plt.figure()
y = conv(x_t(t), u(t), t)
plt.plot(t, y, linewidth=3)
plt.grid()
plt.legend(['y(t)'])
plt.gcf().set_size_inches(6.4, 2.33)

plt.figure()
y = conv(x_t(t), u(t-1), t)
plt.plot(t, y, linewidth=3)
plt.grid()
plt.legend(['y(t)'])
plt.gcf().set_size_inches(6.4, 2.33)