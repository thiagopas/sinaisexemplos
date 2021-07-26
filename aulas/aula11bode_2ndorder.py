from graficos import myplot as plotit
from funcoes import u, window, integrate, conv, movavgc, movavgpast, derivate
import numpy as np
import matplotlib.pyplot as plt
import time

def realpole(w, w0):
    magnitude = u(w) - u(w-w0) + \
                (w0/w) * (u(w-w0))