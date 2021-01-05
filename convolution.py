"""
===============
Embedding in Tk
===============

"""

import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np


def u(t):
    return (t >= -1e-7) * 1.0


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


class Convolver:
    t_s = .1
    t = 0
    x = 0
    h = 0
    y = 0
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

    def __init__(self, t_ini, t_end, t_s=1, delay_inc=.5):
        self.t_s = t_s
        self.t = np.arange(t_ini, t_end, t_s)
        self.y = np.zeros_like(self.t)
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

    def get_delay_samples(self):
        return round(self.delay / self.t_s)

    def get_delay_secs(self):
        return self.delay

    def set_x(self, x_t):
        self.x = np.copy(x_t)

    def set_h(self, h_t):
        self.h = np.copy(h_t)

    def convolve(self):
        self.y = np.zeros_like(self.t)
        y = np.convolve(self.x, self.h) * self.t_s
        valid = np.count_nonzero(self.t < -1e-7)
        self.y[:] = y[valid:self.y.size+valid]
        for i in range(self.t.size - self.t0):
            prod = self.h * self.get_mirror(i)
            max_local = prod.max()
            if max_local > self.prod_max:
                self.prod_max = max_local
            min_local = prod.min()
            if min_local < self.prod_min:
                self.prod_min = min_local
        self.conv_max = self.y.max()
        self.conv_min = self.y.min()

    def get_mirror(self, x_delay=None):
        if x_delay is None:
            x_delay = self.delay
        x_mirror = np.flip(self.x)
        shift = int(np.round(x_delay / self.t_s) + self.t0 - self.t0_mirror)
        x_mirror = np.roll(x_mirror, shift)
        if shift > 0:
            x_mirror[:shift] = 0
        else:
            x_mirror[shift:] = 0
        return x_mirror


c = Convolver(-10, 10, .01, .1)
c.set_x(u(c.t)*c.t)
c.set_h(u(c.t)*c.t)
c.convolve()


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
        c.direction = 'stop'
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
        c.increment()
    elif c.direction == 'left':
        c.decrement()
    else:
        return
    x_shifted = c.get_mirror(c.delay)
    ax1.cla()
    ax1.plot(c.t, c.h, linewidth=2)
    ax1.grid()
    ax2.cla()
    ax2.plot(c.t, x_shifted, linewidth=2)
    ax2.grid()
    ax3.cla()
    ax3.fill_between(c.t, c.h * x_shifted, color='orange')
    ax3.grid()
    ax = ax1.axis()
    ax3.axis([ax[0], ax[1], c.prod_min*1.05, c.prod_max*1.05])
    ax4.cla()
    zero = c.t0 + c.get_delay_samples()
    ax4.plot(c.t[:zero], c.y[:zero])
    ax4.plot(c.t[zero], c.y[zero], 'o', color='orange', linewidth=2)
    ax4.grid()
    ax4.axis([ax[0], ax[1], c.conv_min*1.05, c.conv_max*1.05])
    canvas.draw()


button1 = tkinter.Button(master=root, text="Quit", command=_quit)
button1.pack(side=tkinter.LEFT)

root.after(5, _refreshplot)

tkinter.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.
