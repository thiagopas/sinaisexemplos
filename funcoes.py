import numpy as np


def u(t):
    return (t > 0) + 0.0

def exp_u(t, a):
    return np.exp(-a * t) * ((t > 0) + 0.0)

def window(t, min, max):
    return (t >= min) * (t < max)

def integrate(t, x):
    result = x * (t[1] - t[0])
    for i in np.arange(1, len(result)):
        result[i] = result[i] + result[i-1]
    return result

def sinc(x, eps=1e-10):
    return np.divide(np.sin(x+np.abs(eps)), x+np.abs(eps))

def conv(x, h, t):
    y_full = np.convolve(x, h) * (t[1] - t[0])
    idx = np.count_nonzero(t < 0)
    y = y_full[idx-1:]
    y = y[:len(t)]
    return y

def conv_disc(x, h, t):
    y_full = np.convolve(x, h) * (t[1] - t[0])
    idx = np.count_nonzero(t < 0)
    y = y_full[idx:]
    y = y[:len(t)]
    return y

def movavgc(t, x, T):
    ts = t[1] - t[0]
    t0 = np.argmin(np.abs(t - 0))
    nsamples = np.round(T / ts)
    kernel = np.zeros_like(x)
    kernel[t0-int(nsamples/2):t0+int(nsamples/2)] = 1.0
    y = conv(x, kernel, t)
    return y

def movavgpast(t, x, T):
    ts = t[1] - t[0]
    t0 = np.argmin(np.abs(t - 0))
    nsamples = np.round(T / ts)
    kernel = np.zeros_like(x)
    kernel[t0:t0+int(nsamples)] = 1.0
    y = conv(x, kernel, t)
    return y

def derivate(t, x):
    ts = t[1] - t[0]
    y = np.zeros_like(x)
    y[1:] = np.diff(x) / ts
    return y

def impulse(t):
    ts = t[1] - t[0]
    t0 = np.argmin(np.abs(t))
    x = np.zeros_like(t)
    x[t0] = 1 / ts
    return x