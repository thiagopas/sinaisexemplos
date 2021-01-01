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


root = tkinter.Tk()
root.wm_title("Rocking tank")

fig = Figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)



canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


class WaterTank:
    xpos = np.arange(-3.5, 3.5, .1)
    zeros = np.zeros_like(xpos)
    t_end = 10
    t_s = .1
    delays = np.arange(0, t_end, t_s)
    samples = 5
    fir = np.ones(samples) / samples
    offset = 3
    plot = np.ones_like(delays) * offset
    product = np.ones_like(delays) * offset
    filtered = np.copy(plot)
    f_osc = .25
    angle_inc = f_osc*2*np.pi*t_s
    angle = 0


wt = WaterTank()


def sin(pos):
    return np.sin(.5 * np.pi * pos / (wt.xpos[-1]))


def levelfunc(pos, angle, offset):
    a = np.arctan(np.sin(angle)*5) / 4
    return pos * a + offset + a * 2 * sin(pos)


def drawtank(angle, offset):
    ax1.cla()
    level = levelfunc(wt.xpos, angle, offset)
    top = 7
    ax1.fill_between(wt.xpos, level, wt.zeros)
    linewidth=6
    ax1.plot([wt.xpos[0], wt.xpos[-1]], [0, 0], 'k', linewidth=linewidth)
    ax1.plot([wt.xpos[0], wt.xpos[0]], [0, top], 'k', linewidth=linewidth)
    ax1.plot([wt.xpos[-1], wt.xpos[-1]], [0, top], 'k', linewidth=linewidth)
    ax1.plot([wt.xpos[-1]], [level[-1]], 'ro')
    ax1.plot([wt.xpos[0], wt.xpos[-1]], [wt.offset, wt.offset], 'k:')
    ax1.axis('square')


def plot_level_time():
    ax = ax1.axis()
    ax2.cla()
    ax2.plot(wt.delays, wt.plot)
    ax2.plot([0], [wt.plot[0]], 'ro')
    ax2.plot(wt.delays, wt.filtered, 'k')
    ax2.axis('square')
    ax2.axis([0, ax[1]-ax[0], ax[2], ax[3]])
    ax2.grid()
    ax2.set_title('Historic plot')

def plot_fir():
    ax3.cla()
    fir_padded = np.zeros(wt.delays.size)
    fir_padded[:wt.fir.size] = wt.fir
    ax3.plot(wt.delays, fir_padded)
    #ax3.axis('square')
    ax = ax2.axis()
    ax3.axis([ax[0], ax[1], 0, 1.2])
    ax3.grid()
    ax3.set_title('Impulse response')


def plot_conv():
    ax4.cla()
    ax4.fill_between(wt.delays[:wt.product.size], wt.product, color='black')
    #ax4.axis('square')
    ax = ax2.axis()
    ax4.axis([ax[0], ax[1], 0, 1.2])
    ax4.grid()
    ax4.set_title('Product')


def fir_inc():
    wt.samples += 1
    wt.fir = np.ones(wt.samples) / wt.samples


def fir_dec():
    if wt.samples > 1:
        wt.samples -= 1
        wt.fir = np.ones(wt.samples) / wt.samples


def on_key_press(event):
    #print(format(event.key))
    if event.key == 'right':
        fir_inc()
        plot_fir()
    elif event.key == 'left':
        fir_dec()
        plot_fir()
    elif event.key == 'up':
        wt.offset += .2
    elif event.key == 'down':
        wt.offset -= .2



canvas.mpl_connect("key_press_event", on_key_press)


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


def _refreshplot():
    root.after(int(wt.t_s * 1000), _refreshplot)
    wt.angle += wt.angle_inc
    drawtank(wt.angle, wt.offset)
    wt.plot = np.roll(wt.plot, 1)
    wt.plot[0] = levelfunc(wt.xpos[-1], wt.angle, wt.offset)
    wt.filtered = np.roll(wt.filtered, 1)
    wt.product = wt.fir * wt.plot[:wt.fir.size]
    wt.filtered[0] = np.sum(wt.product)
    plot_level_time()
    plot_fir()
    plot_conv()
    canvas.draw()


button1 = tkinter.Button(master=root, text="Quit", command=_quit)
button1.pack(side=tkinter.LEFT)

root.after(500, _refreshplot)

tkinter.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.
