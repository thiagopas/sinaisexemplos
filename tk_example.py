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
ax1 = fig.add_subplot(111)



canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)



def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


def _refreshplot():
    root.after(int(500), _refreshplot)

    ################## INÍCIO DO CÓDIGO CÍCLICO ####################
    ax1.cla()
    vetor = np.random.random(10)
    ax1.plot(vetor)
    ################### FIM DO CÓDIGO CÍCLICO   ####################


    canvas.draw()


root.after(500, _refreshplot)

tkinter.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.
