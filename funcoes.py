import numpy as np


def u(t):
    return (t >= 0)

def window(t, min, max):
    return (t >= min) * (t < max)

