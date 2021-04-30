from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate, impulse
import numpy as np
import matplotlib.pyplot as plt

def piyaxis(ax):
    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(["{:.3f}".format(y / np.pi) + 'π' for y in ticks_loc])

# Questao 2b
sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
w = np.arange(-30, 30, .1)
xtick = np.arange(w[0], w[-1], 5)
ytick = np.linspace(0, np.pi/20, 3)
resp = (np.pi/20) * (u(w+20)-u(w+15)+u(w-15)-u(w-20))
axis = [w[0], w[-1], -.1*np.max(resp), 1.1*np.max(resp)]
plotit(x=w, y=resp, ax=ax_x, axis=axis, xticks=xtick, yticks=ytick, xlabel='$\omega$', title='$Y(\omega)$')
piyaxis(ax_x)