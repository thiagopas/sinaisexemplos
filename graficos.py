import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time


def myplot(xticks=None, yticks=None, x=[], y=[], save=False,
           axis=None, ax=None, wait=False, title=None, dirac=None,
           xlabel='', ylabel='', legend=None, text=None, linewidth=3):
    if wait is True:
        plt.pause(.01)
        plt.waitforbuttonpress()
    if ax is None:
        obj = plt.subplots(1, 1)[1]
    else:
        obj = ax
    obj.cla()
    if title is not None:
        obj.set_title(title)
    obj.plot([-1e10, 1e10], [0, 0], 'k', linewidth=linewidth)
    obj.plot([0, 0], [-1e10, 1e10], 'k', linewidth=linewidth)
    if dirac is None:
        obj.plot(x, y, 'blue', linewidth=linewidth)
    else:
        y_ = np.copy(y)
        for i in range(len(dirac[0])):
            t_pulse = np.argmin(np.abs(x - dirac[0][i]))
            y_[t_pulse] = (y[t_pulse-1] + y[t_pulse+1]) / 2
            arrow = mpl.patches.FancyArrowPatch(posA=(dirac[0][i], 0), posB=(dirac[0][i], dirac[1][i]),
                                                arrowstyle='-|>', mutation_scale=30,
                                                linewidth=linewidth, color='blue')
            obj.add_patch(arrow)
            obj.plot([dirac[0][i], dirac[0][i]], [0, dirac[1][i]], 'blue', linewidth=linewidth)
        obj.plot(x, y_, 'blue', linewidth=linewidth)
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
    if wait is True or save is True:
        plt.savefig('../temp/' + time().__str__() + '.png')