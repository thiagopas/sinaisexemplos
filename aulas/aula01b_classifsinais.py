from graficos import myplot as plotit
from funcoes import u, window
from scipy.signal import step, lti
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

plt.figure()
t = np.arange(0, 10, .01)
x = 3*(np.cos(t) + np.cos(2*t))
plt.plot(t, x, linewidth=2)
plt.xticks(np.arange(0, 11))
plt.yticks(np.arange(-4, 7))
plt.grid()

plt.figure()
t_disc = t[np.mod(t, 1) == 0]
x_disc = x[np.mod(t, 1) == 0]
np.mod(t, 1) == 0
markerline, stemlines, baseline = plt.stem(t_disc, x_disc)
plt.setp(stemlines, color='C0', linewidth=2)
plt.setp(baseline, color='k', linewidth=2)
plt.show()
plt.xticks(np.arange(0, 11))
plt.yticks(np.arange(-4, 7))
plt.grid()

plt.figure()
t = np.arange(0, 10, .01)
x = 3*(np.cos(t) + np.cos(2*t))
plt.plot(t, np.round(x), linewidth=2)
plt.xticks(np.arange(0, 11))
plt.yticks(np.arange(-4, 7))
plt.grid()

plt.figure()
t_disc = t[np.mod(t, 1) == 0]
x_disc = x[np.mod(t, 1) == 0]
np.mod(t, 1) == 0
markerline, stemlines, baseline = plt.stem(t_disc, np.round(x_disc))
plt.setp(stemlines, color='C0', linewidth=2)
plt.setp(baseline, color='k', linewidth=2)
plt.show()
plt.xticks(np.arange(0, 11))
plt.yticks(np.arange(-4, 7))
plt.grid()

# Sinais aleatórios
sp = plt.subplots(1, 2)
ax_time = sp[0].axes[0]
ax_freq = sp[0].axes[1]
data = scipy.io.wavfile.read('C:/Users/thiag/Downloads/test.wav')
t = np.arange(0, len(data[1])) / float(data[0])
voice = data[1][:, 0] / 10000
ax_time.plot(t, voice)
ax_time.set_xlabel('t [s]')
voice_f = np.fft.fft(voice)
f = np.linspace(0, data[0], len(t))
f = f - f.max() / 2
f = f / 1000
ax_freq.plot(f, np.abs(np.fft.fftshift(voice_f)))
ax_freq.set_xlabel('f [kHz]')
ax_freq.axis([-10, 10, -263.30541040742713, 5529.415644975601])

plt.figure()
t = np.arange(0, 2, 0.005)
x = np.random.randn(len(t))
plt.plot(t, x)

# Sinal par e sinal ímpar
sp = plt.subplots(2, 1)
ax_x = sp[0].axes[0]
ax_y = sp[0].axes[1]
t = np.arange(-5, 5, .01)
x = np.cos(t)
y = np.abs(t)
plotit(ax=ax_x, x=t, y=x, axis=[t[0], t[-1], -1.1, 1.1], wait=False)
plotit(ax=ax_y, x=t, y=y, axis=[t[0], t[-1], -5.1, 5.1], wait=False)

sp = plt.subplots(2, 1)
ax_x = sp[0].axes[0]
ax_y = sp[0].axes[1]
t = np.arange(-5, 5, .01)
x = np.sin(t)
y = t
plotit(ax=ax_x, x=t, y=x, axis=[t[0], t[-1], -1.1, 1.1], wait=False)
plotit(ax=ax_y, x=t, y=y, axis=[t[0], t[-1], -5.1, 5.1], wait=False)

# Exponencial complexa
plt.figure()
x = np.exp(1j*t)
plt.plot(t, np.real(x), linewidth=2)
plt.plot(t, np.imag(x), linewidth=2)
plt.legend(['Re{x(t)}', 'Im{x(t)}'])
plt.grid()

# Periódico
sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
t = np.arange(-2, 8, .001)
x = np.mod(t, 2) - 1
plotit(ax=ax_x, x=t, y=x, axis=[t[0], t[-1], -1.2, 1.2], wait=False, linewidth=2)
plt.gcf().set_size_inches([7.19, 3.25])

sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
x = np.mod(t, 2) - 1
x[(3.8 <= t) * (t <= 4)] = x[np.argmin(np.abs(t-3.8))]
plotit(ax=ax_x, x=t, y=x, axis=[t[0], t[-1], -1.2, 1.2], wait=False, linewidth=2)
plt.gcf().set_size_inches([7.19, 3.25])

# Energia e potência
plt.figure()
t = np.arange(-5, 5, .001)
x = 5 * np.exp(2j*t)
plt.plot(t, np.real(x), linewidth=2)
plt.plot(t, np.imag(x), linewidth=2)
plt.legend(['Re{x(t)}', 'Im{x(t)}'])
plt.grid()

sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
t = np.arange(-2.2, 1.8, .001)
x = np.array(np.mod(t, 1) < .5, dtype=float)
plotit(ax=ax_x, x=t, y=x, axis=[t[0], t[-1], -.2, 1.2], wait=False, linewidth=2)

sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
t = np.arange(-2.2, 1.8, .001)
x = np.exp(t)
plotit(ax=ax_x, x=t, y=x, axis=[t[0], t[-1], -.5, 6], wait=False, linewidth=2)

# Causais e não causais
sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
axis = [-1, 2.7, -.2, .8]
t = np.arange(-1, 5, .01)
x = u(t)
sys = lti([20], [1, 5, 40])
t_step, y = step(sys, T=t*(t>=0))
plotit(ax=ax_x, x=t_step, y=y, axis=axis, wait=False, hold=True)
plotit(ax=ax_x, x=[t[0], 0], y=[0, 0], axis=axis, wait=False, hold=True)
plt.gcf().set_size_inches(6.4 , 3.05)

sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
plotit(ax=ax_x, x=t[:-50], y=y[50:], axis=axis, wait=False, hold=True)
plt.gcf().set_size_inches(6.4, 3.05)

# Quizz: analógico / digital
plt.figure()
plt.subplot(1, 2, 1)
t = np.arange(-2, 5, .01)
x = 5 * np.exp(-(.4*t)**2)
plt.plot(t, x, linewidth=2)
plt.xticks(np.arange(-2, 6))
plt.yticks(np.arange(-1, 7))
plt.grid()
plt.legend(['x(t)'])

plt.subplot(1, 2, 2)
plt.plot(t, np.round(x), linewidth=2)
plt.xticks(np.arange(-2, 6))
plt.yticks(np.arange(-1, 7))
plt.grid()
plt.legend(['y(t)'])


# Quizz: Contínuo / discreto
plt.figure()
plt.subplot(1, 2, 1)
t = np.arange(-2, 5.1, .01)
x = 5 * np.exp(-(.4*t)**2)
plt.plot(t, x, linewidth=2)
plt.xticks(np.arange(-2, 6))
plt.yticks(np.arange(-1, 7))
plt.grid()
plt.legend(['x(t)'])

plt.subplot(1, 2, 2)
t_disc = t[np.abs(np.mod(t, 1)) < .01]
x_disc = x[np.abs(np.mod(t, 1)) < .01]
np.mod(t, 1) == 0
markerline, stemlines, baseline = plt.stem(t_disc, x_disc)
plt.setp(stemlines, color='C0', linewidth=2)
plt.setp(baseline, color='k', linewidth=2)
plt.show()
plt.xticks(np.arange(-2, 6))
plt.yticks(np.arange(-1, 7))
plt.grid()
plt.legend(['x[n]'])

# Questão APS: exp(-abs(t))
sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
t = np.arange(-7, 7, .01)
x = np.exp(-np.abs(t))
plotit(ax=ax_x, x=t, y=x, axis=[t[0], t[-1], -.2, 1.2], wait=False, linewidth=2, xlabel='t [s]', ylabel='x(t)')