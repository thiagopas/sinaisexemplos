import numpy as np
import matplotlib.pyplot as plt

def myplot(xticks=None, yticks=None, x=[], y=[],
           axis=None,
           xlabel='', ylabel=''):
    plt.figure()
    plt.plot([-1e10, 1e10], [0, 0], 'k', linewidth=2)
    plt.plot([0, 0], [-1e10, 1e10], 'k', linewidth=2)
    plt.plot(x, y, 'blue', linewidth=4)
    plt.grid()
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if axis is not None:
        plt.axis(axis)
    else:
        plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
        plt.axis('equal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)