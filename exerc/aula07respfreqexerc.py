import numpy as np
import random

f = open('../temp/exp/RA.txt')
fo = open('../temp/exp/RArand.txt', 'w')
lines = f.readlines()
t = np.arange(-15, 15, 1e-3)
for i in range(len(lines)):
    ra = lines[i].replace('\n', '')
    A = 2.5 * random.random()
    phi = 2 * np.pi * random.random() - np.pi
    omega = 3 * random.random()
    x = A * np.exp(1j * (omega * t + phi))
    np.save('../temp/exp/' + ra + '.npy', x)
    fo.write(ra + '\t' + str(A) + '\t' + str(phi) + '\t' + str(omega) + '\n')
fo.close()
f.close()