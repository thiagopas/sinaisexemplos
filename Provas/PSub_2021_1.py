import numpy as np
import matplotlib.pyplot as plt
from funcoes import conv_disc
from graficos import myplot as plotit
from graficos import pixaxis, piyaxis

def ustep(t):
    result = (t >= 0)
    return np.array(result, dtype=float)

# Questão 1b
sp = plt.subplots(1, 1)
ax_x = sp[0].axes[0]
w = np.arange(-100*np.pi, 100*np.pi, .1)
xtick = np.array([-60, 0, 60], dtype=float)*np.pi
ytick = np.array([0, 1])*np.pi
y = np.zeros_like(w)
axis = [-100*np.pi, 100*np.pi, -.2, np.pi*1.1]
dirac = [[-60*np.pi, 60*np.pi], [np.pi, np.pi]]
plotit(x=w, y=y, ax=ax_x, axis=axis, xticks=xtick, dirac=dirac, yticks=ytick, xlabel='ω', ylabel='|Y(ω)|')
pixaxis(ax_x)
piyaxis(ax_x)

#Questão 2a
n = np.arange(-2, 6)
y = np.array(n == 0, dtype=float) * 1 + np.array(n >= 1, dtype=float) * 2
plt.figure()
plt.stem(n, y)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid()

#Questão 2b
n = np.arange(-2, 6)
y = np.array(n >= 0, dtype=float) * (n + 1)
plt.figure()
plt.stem(n, y)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid()