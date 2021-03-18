import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime


class ImgCounter:
    img_counter = 0

def piyaxis(ax):
    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(["{:.1f}".format(y / np.pi) + 'π' for y in ticks_loc])

def pixaxis(ax):
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels(["{:.0f}".format(x / np.pi) + 'π' for x in ticks_loc])

def myplot(xticks=None, yticks=None, x=[], y=[], save=False, color='blue',
           axis=None, ax=None, wait=False, title=None, dirac=None, hold=False,
           xlabel='', ylabel='', legend=None, text=None, linewidth=2, axes=True):
    if wait is True:
        plt.pause(.01)
        plt.waitforbuttonpress()
    if ax is None:
        obj = plt.subplots(1, 1)[1]
    else:
        obj = ax
    if hold is False:
        obj.cla()
    if title is not None:
        obj.set_title(title)
    if axes is not False:
        obj.plot([-1e10, 1e10], [0, 0], 'k', linewidth=2)
        obj.plot([0, 0], [-1e10, 1e10], 'k', linewidth=2)
    if dirac is None:
        obj.plot(x, y, color, linewidth=linewidth)
    else:
        y_ = np.copy(y)
        for i in range(len(dirac[0])):
            t_pulse = np.argmin(np.abs(x - dirac[0][i]))
            y_[t_pulse] = (y[t_pulse-1] + y[t_pulse+1]) / 2
            arrow = mpl.patches.FancyArrowPatch(posA=(dirac[0][i], 0), posB=(dirac[0][i], dirac[1][i]),
                                                arrowstyle='-|>', mutation_scale=30,
                                                linewidth=linewidth, color=color)
            obj.add_patch(arrow)
            obj.plot([dirac[0][i], dirac[0][i]], [0, dirac[1][i]], color, linewidth=linewidth)
        obj.plot(x, y_, color, linewidth=linewidth)
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
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        imgcounter = ImgCounter.img_counter
        ImgCounter.img_counter += 1
        plt.savefig('../temp/' + time_str + ' ' + str(imgcounter).zfill(3) + '.png')