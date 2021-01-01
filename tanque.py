import matplotlib.pyplot as plt
import numpy as np

xpos = np.arange(-3.5, 3.5, .1)
zeros = np.zeros_like(xpos)
t_end = 10
t_s = .1
delays = np.arange(0, t_end, t_s)
fig, (ax1, ax2) = plt.subplots(1, 2)
samples = 40
fir = np.ones(samples) / samples
offset = 3
plot = np.ones_like(delays) * offset
filtered = np.copy(plot)

def sin(pos):
    return np.sin(.5 * np.pi * pos / (xpos[-1]))


def levelfunc(pos, angle, offset):
    a = np.arctan(np.sin(angle)*5) / 4
    return pos * a + offset + a * 2 * sin(pos)


def drawtank(angle, offset):
    ax1.cla()
    level = levelfunc(xpos, angle, offset)
    top = 7
    ax1.fill_between(xpos, level, zeros)
    linewidth=6
    ax1.plot([xpos[0], xpos[-1]], [0, 0], 'k', linewidth=linewidth)
    ax1.plot([xpos[0], xpos[0]], [0, top], 'k', linewidth=linewidth)
    ax1.plot([xpos[-1], xpos[-1]], [0, top], 'k', linewidth=linewidth)
    ax1.plot([xpos[-1]], [level[-1]], 'ro')
    ax1.axis('square')


def plot_level_time():
    ax = ax1.axis()
    ax2.cla()
    ax2.plot(delays, plot)
    ax2.plot([0], [plot[0]], 'ro')
    ax2.plot(delays, filtered, 'k')
    ax2.axis('square')
    ax2.axis([0, ax[1]-ax[0], ax[2], ax[3]])
    ax2.grid()


angles = np.arange(0, 2*np.pi, .2)
cycles = 6
angles_r = np.zeros(len(angles)*cycles)
for i in range(cycles):
    angles_r[i*len(angles):(i+1)*len(angles)] = angles


for i in range(len(angles_r)):
    drawtank(angles_r[i], offset)
    plot = np.roll(plot, 1)
    plot[0] = levelfunc(xpos[-1], angles_r[i], offset)
    filtered = np.roll(filtered, 1)
    filtered[0] = np.sum(fir * plot[:fir.size])
    plot_level_time()
    plt.show()
    plt.pause(.001)

