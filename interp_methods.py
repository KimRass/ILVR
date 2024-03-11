import math
import numpy as np


def support_sz(sz):
    def wrapper(f):
        f.support_sz = sz
        return f
    return wrapper


@support_sz(4)
def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (
        (1.5 * absx3 - 2.5 * absx2 + 1.) * (absx <= 1.) +(
            -0.5 * absx3 + 2.5 * absx2 - 4. * absx + 2.
        ) * ((1. < absx) & (absx <= 2.))
    )


@support_sz(4)
def lanczos2(x):
    eps = np.finfo(np.float32).eps
    return ((np.sin(math.pi * x) * np.sin(math.pi * x / 2) + eps) / (
            (math.pi ** 2 * x ** 2 / 2) + eps
        )) * (abs(x) < 2)


@support_sz(6)
def lanczos3(x):
    eps = np.finfo(np.float32).eps
    return ((np.sin(math.pi * x) * np.sin(math.pi * x / 3) + eps) / (
            (math.pi ** 2 * x ** 2 / 3) + eps
        )) * (abs(x) < 3)


@support_sz(2)
def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


@support_sz(1)
def box(x):
    return ((-1 <= x) & (x < 0)) + ((0 <= x) & (x <= 1))
