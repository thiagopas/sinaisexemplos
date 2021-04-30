from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
x = np.outer(np.linspace(-1, 4, 300), np.ones(300))
y = x.copy().T # transpose
z = 1 / (np.abs((x+1j*y - 2)) + 1e-2)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none', vmax=3)
ax.set_title('|H(s)|')
#y_f = y[0, :]
#x_f = np.zeros_like(y_f)
#z_f = 1 / (np.abs((1j*y_f - 2)))
#ax.plot3D(x_f, y_f, z_f, linewidth=3, color='C1', alpha=1)
ax.set_zlim([0, 3])
ax.set_xlabel('$\sigma$')
ax.set_ylabel('$j\omega$')
plt.show()

plt.figure()
plt.imshow(np.flip(np.log10(z).transpose(), 0), vmax=1.5, extent=[x[0,0], x[-1,0], y[0,0], y[0,-1]])
plt.xlabel('$\sigma$')
plt.ylabel('$j\omega$')
plt.colorbar()