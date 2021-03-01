from graficos import myplot as plotit
from funcoes import u, window
import numpy as np
import matplotlib.pyplot as plt
import time

def xfunc(t):
    x = np.sin(t + .418) + t/4 + 1
    return x

def x_sample(x, t, T):
    sample = np.zeros_like(T, dtype=float)
    for i in range(len(T)):
        idx0 = np.argmin(np.abs(t - T[i]))
        idx1 = np.argmin(np.abs(t - T[i] + .5))
        sample[i] = np.sum(x[idx1:idx0]) / (idx0 - idx1)
    return sample

def x_sample_ideal(x, t, T):
    sample = np.zeros_like(T, dtype=float)
    for i in range(len(T)):
        idx0 = np.argmin(np.abs(t - T[i]))
        sample[i] = x[idx0]
    return sample

def x_exerc1(t):
    return np.power(np.sqrt(np.abs(t)) - 1, 2)

t = np.arange(-6, 6, .01)
x = xfunc(t)
sp = plt.subplots(1,1)[1]
sp.plot(t, x)
plt.grid()
sp.figure.set_figwidth(6.4)
sp.figure.set_figheight(3.0)
plt.savefig('../temp/' + str(time.time()) + '.png')
T = np.arange(-6, 8, 2)
samples = x_sample(x, t, T)
plt.stem(T, samples, 'r')
plt.savefig('../temp/' + str(time.time()) + '.png')

sp = plt.subplots(1,1)[1]
sp.plot(t, x)
plt.grid()
sp.figure.set_figwidth(6.4)
sp.figure.set_figheight(3.0)
plt.savefig('../temp/' + str(time.time()) + '.png')
T = np.arange(-6, 8, 2)
samples = x_sample_ideal(x, t, T)
plt.stem(T, samples, 'r')
plt.savefig('../temp/' + str(time.time()) + '.png')

sp = plt.subplots(1,1)[1]
sp.plot(t, x)
plt.grid()
sp.figure.set_figwidth(6.4)
sp.figure.set_figheight(3.0)
plt.savefig('../temp/' + str(time.time()) + '.png')

x = x_exerc1(t)
plotit(x=t, y=x, axis=[-1, 6, -.5, 2.5])