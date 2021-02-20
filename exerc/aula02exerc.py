import graficos as g
from funcoes import u, window
import numpy as np
import matplotlib.pyplot as plt

def x(t):
    return (u(t)*1.0 - u(t-3)*1.0)*np.sin(t*np.pi/3)

t = np.arange(-5, 5, 0.01)
plt.figure()
plt.plot(t, x(t))
plt.grid()
plt.figure()
plt.plot(t, x(-t))
plt.grid()
plt.figure()
plt.plot(t, x(-t+2.0))
plt.grid()