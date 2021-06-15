from graficos import myplot as plotit
from funcoes import u, window
from scipy.signal import step, lti
import numpy as np
import matplotlib.pyplot as plt

sp = plt.subplots(2, 1)
ax_x = sp[0].axes[0]
ax_y = sp[0].axes[1]
axis = [-1, 2.7, -.2, 1.2]
t = np.arange(-1, 2.7, .01)
x = u(t)
y = .5*x
plotit(ax=ax_x, x=t, y=x, axis=axis, wait=False)
plotit(ax=ax_y, x=t, y=y, axis=axis, wait=False)

sp = plt.subplots(2, 1)
ax_x = sp[0].axes[0]
ax_y = sp[0].axes[1]
axis = [-1, 2.7, -.2, 1.2]
t = np.arange(-1, 2.7, .01)
x = u(t)
sys = lti([20], [1, 5, 40])
t_step, y = step(sys)
plotit(ax=ax_x, x=t, y=x, axis=axis, wait=False)
plotit(ax=ax_y, x=t_step, y=y, axis=axis, wait=False, hold=True)
plotit(ax=ax_y, x=[t[0], 0], y=[0, 0], axis=axis, wait=False, hold=True)
