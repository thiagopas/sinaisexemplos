import graficos as g
from funcoes import u, window
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(-4, 4, .01)
x1 = (t*3/2 + 3)*window(t, -2, 0) + (-t*3/2 + 3)*window(t, 0, 2)
g.myplot(x=t, y=x1, axis=[-4, 4, -1, 4])
plt.savefig('../temp/01_x1')

x2 = 2*window(t, -3, 3)
g.myplot(x=t, y=x2, axis=[-4, 4, -1, 4])
plt.savefig('../temp/02_x2')

x3 = 5*t*window(t, 0, 1) + 5*window(t, 1, 2) + (15-5*t)*window(t, 2, 3)
g.myplot(x=t, y=x3, axis=[-4, 4, -1, 5.5])
plt.savefig('../temp/03_x3')

x4 = 2*t*window(t, 0, 1.5)
g.myplot(x=t, y=x4, axis=[-4, 4, -1, 4])
plt.savefig('../temp/04_x4')

x4e = 2*(t*.5)*window(t*.5, 0, 1.5)
g.myplot(x=t, y=x4e, axis=[-4, 4, -1, 4])
plt.savefig('../temp/05_x4e')

x4ed = 2*((t+2)*.5)*window((t+2)*.5, 0, 1.5)
g.myplot(x=t, y=x4ed, axis=[-4, 4, -1, 4])
plt.savefig('../temp/06_x4ed')

x4d = 2*(t+2)*window(t+2, 0, 1.5)
g.myplot(x=t, y=x4d, axis=[-4, 4, -1, 4])
plt.savefig('../temp/07_x4d')

x4de = 2*(t*.5+2)*window(t*.5+2, 0, 1.5)
g.myplot(x=t, y=x4de, axis=[-4, 4, -1, 4])
plt.savefig('../temp/08_x4de')

t = np.arange(-1, 9, 0.01)
periodico = (np.mod(t, 2) - 1)*3
g.myplot(x=t, y=periodico, axis=[-1, 9, -3.5, 3.5])
plt.gcf().set_figheight(2.93)
plt.savefig('../temp/09_periodico')