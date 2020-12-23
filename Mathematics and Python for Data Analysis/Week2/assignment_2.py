from matplotlib import pylab
import pandas as pd
import numpy as np
import math
from scipy.linalg import solve


# Investigation function
def f(x):
    return math.sin(x / 5) * math.exp(x / 10) + 5 * math.exp(-x / 2)


# Polynomial definition:
# f(x) = W0 + (W1 * x^1) + (W2 * x^2) +...+(Wn * x^n)
# First degree polynomials (linear polynomial):
# f(x) = W0 + W1 * x
# Create linear polynomial, which based on values in x = 1 and x = 15
# Current example:
# f(x1) = W0 + W1 * x1 =>  f_1 = W0 + W1
# f(x2) = W0 + W1 * x2 => f_15 = W0 + W1 * 15
# So, there 2 equations with 2 unknowns => can be resolved
f_1 = f(1)
f_15 = f(15)
x = np.array([[1, 1], [1, 15]])
b = np.array([f_1, f_15])
w_1 = solve(x, b)


def func_polynomial_n_1(var_x):
    return w_1[0] + w_1[1] * var_x


# calculate polynomial n=2
# form:
# f(x) = W0 + W1 * x + W2 * x^2
# our example:
# f_1 = W0 + W1    + W2
# f_8 = W0 + W1*8  + W2*64
# f_15= W0 + W1*15 + W2*225
f_1 = f(1)
f_8 = f(8)
f_15 = f(15)

x = np.array([[1, 1, 1], [1, 8, 64], [1, 15, 225]])
y = np.array([f_1, f_8, f_15])
w_2 = solve(x, y)


def func_polynomial_n_2(var_x):
    return w_2[0] + w_2[1] * var_x + w_2[2] * math.pow(var_x, 2)


# Show on graph f, n=1, n=2
x = np.arange(16)
y = map(func_polynomial_n_2, x)
pylab.plot(x, list(y))

y = map(func_polynomial_n_1, x)
pylab.plot(x, list(y))

y = map(f, x)
pylab.plot(x, list(y))

pylab.show()
