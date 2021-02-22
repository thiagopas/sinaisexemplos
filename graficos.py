import numpy as np
import matplotlib.pyplot as plt
from time import time

def myplot(xticks=None, yticks=None, x=[], y=[],
           axis=None, ax=None, wait=False, title=None,
           xlabel='', ylabel='', legend=None, text=None):
    if wait is True:
        plt.pause(.01)
        plt.waitforbuttonpress()
    if ax is None:
        obj = plt.subplots(1, 1)[0][0]
    else:
        obj = ax
    obj.cla()
    if title is not None:
        obj.set_title(title)
    obj.plot([-1e10, 1e10], [0, 0], 'k', linewidth=2)
    obj.plot([0, 0], [-1e10, 1e10], 'k', linewidth=2)
    obj.plot(x, y, 'blue', linewidth=4)
    obj.grid(True)
    if xticks is not None:
        obj.set_xticks(xticks)
    if yticks is not None:
        obj.set_yticks(yticks)
    if axis is not None:
        obj.axis(axis)
    elif len(x) > 0:
        obj.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
        obj.axis('equal')
    obj.set_xlabel(xlabel)
    h = obj.set_ylabel(ylabel)
    if legend is not None:
        obj.legend(legend, prop={'size': 15})
    if text is not None:
        obj.text(.2, 2.2, text, fontsize=15)
    h.set_fontsize(15)
    h.set_rotation(0)
    plt.draw()
    plt.show()
    if wait is True:
        plt.savefig('../temp/' + time().__str__() + '.png')