
import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
from funcoes import u, conv

def gaussian(t, avg, std):
    x = np.exp(-.5*((t-avg)/std)**2)
    thewindow = np.copy(x[np.where(np.array(t >= 3, dtype=int) * np.array(t <= 3.7, dtype=int))])
    line = np.zeros_like(thewindow)
    step = (1.5 - thewindow[0]) / len(thewindow)
    line = np.arange(thewindow[0], 1.5, step)
    x[np.where(np.array(t >= 3, dtype=int) * np.array(t <= 3.7, dtype=int))] = line
    return x

def dummy(x):
    return ''

def eraselabels(ax):
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels([dummy(x) for x in ticks_loc])

def h_def(t):
    R = 2
    L = 1
    return np.exp(-t*R/L)*u(t) / L

root = tkinter.Tk()
root.wm_title("Convolution")

fig = Figure()
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


tmin = -10
tmax = 10

class Convolver:
    t_s = .1
    t = 0
    h_func = None
    x = 0
    y = 0
    h = 0
    product = 0
    t0 = 0
    t0_mirror = 0
    direction = ''
    delay = 0
    delay_inc = .1
    prod_min = 0
    prod_max = 0
    conv_min = 0
    conv_max = 0

    def __init__(self, x, h_func, t, delay_inc=.5):
        self.t_s = t[1] - t[0]
        self.t = t
        self.x = x
        self.h_func = h_func
        self.h = h_func(t)
        self.y = conv(x, self.h, t)
        self.product = np.zeros_like(self.t)
        self.t0 = np.argmin(np.abs(self.t))
        self.t0_mirror = np.argmin(np.abs(np.flip(self.t)))
        self.direction = 'stop'
        self.delay = 0
        self.delay_inc = delay_inc


    def increment(self):
        self.delay += self.delay_inc

    def decrement(self):
        self.delay -= self.delay_inc

    def reset(self):
        self.delay = 0

    def get_delay_samples(self):
        return round(self.delay / self.t_s)

    def get_delay_secs(self):
        return self.delay

    def get_product(self):
        return self.x * self.h_func(-t + self.delay)

    def get_y_trunc(self):
        idx_now = np.argmin(np.abs(t - self.delay))
        t_trunc = self.t[:idx_now]
        y_trunc = self.y[:idx_now]
        return np.array([t_trunc, y_trunc])


t = np.arange(-10, 10, .01)
x = gaussian(t, 2, 1.5)
c = Convolver(x, u, t, .1)
tmin = -5
tmax = 5
xticks = np.arange(-10, 10, 1)


def on_key_press(event):
    # print(format(event.key))
    if event.key == 'right':
        if c.direction == 'stop':
            c.direction = 'right'
        else:
            c.direction = 'stop'
    elif event.key == 'left':
        if c.direction == 'stop':
            c.direction = 'left'
        else:
            c.direction = 'stop'
    elif event.key == 'up':
        c.direction = 'clear'
        c.reset()
    elif event.key == 'down':
        c.direction = 'stop'


canvas.mpl_connect("key_press_event", on_key_press)


def _quit():
    root.quit()  # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


def _refreshplot():
    root.after(1, _refreshplot)
    if c.direction == 'right':
        c.direction = 'stop'
        c.increment()
    elif c.direction == 'left':
        c.direction = 'stop'
        c.decrement()
    elif c.direction == 'clear':
        c.direction = 'stop'
    else:
        return
    ax1.cla()
    ax1.plot(c.t, c.x, linewidth=2)
    ax1.legend(['x(τ)'])
    ax1.grid()
    ax1.axis([tmin, tmax, np.min(x) - .1*(np.max(x)-np.min(x)),
              np.max(x) + .1*(np.max(x)-np.min(x))])
    ax1.set_xticks(xticks)
    eraselabels(ax1)
    ax2.cla()
    ax2.plot(c.t, c.h_func(-c.t + c.delay), linewidth=2)
    ax2.legend(['h(' + "{:.1f}".format(c.delay) + '-τ)'])
    ax2.axis([tmin, tmax, np.min(c.h) - .1 * (np.max(c.h) - np.min(c.h)),
              np.max(c.h) + .1 * (np.max(c.h) - np.min(c.h))])
    ax2.grid()
    ax2.set_xticks(xticks)
    eraselabels(ax2)
    ax3.cla()
    ax3.plot(c.t, c.get_product())
    ax3.legend(['x(τ)h(' + "{:.1f}".format(c.delay) + '-τ)'])
    ax3.fill_between(c.t, c.get_product(), color='orange')
    ax3.grid()
    prodmax = np.max([np.max(c.x)*np.max(c.h), np.min(c.x)*np.min(c.h)])
    prodmin = np.min([np.max(c.x)*np.min(c.h), np.min(c.x)*np.max(c.h)])
    ax3.axis([tmin, tmax, prodmin - .1 * (prodmax - prodmin),
              prodmax + .1 * (prodmax - prodmin)])
    ax3.set_xticks(xticks)
    eraselabels(ax3)
    ax4.cla()
    trunc = c.get_y_trunc()
    ax4.plot(trunc[0], trunc[1])
    if trunc[0].size > 0:
        ax4.plot(trunc[0][-1], trunc[1][-1], 'o', color='orange', linewidth=2)
        ax4.legend(['y(t)', 'y(' + "{:.1f}".format(c.delay) + ')=' + "{:.3f}".format(trunc[1][-1])])
    else:
        ax4.legend(['y(t)'])
    ax4.grid()
    ax4.axis([tmin, tmax, np.min(c.y) - .1 * (np.max(c.y) - np.min(c.y)),
              np.max(c.y) + .1 * (np.max(c.y) - np.min(c.y))])
    ax4.set_xticks(xticks)
    canvas.draw()


button1 = tkinter.Button(master=root, text="Quit", command=_quit)
button1.pack(side=tkinter.LEFT)

root.after(5, _refreshplot)

tkinter.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.
