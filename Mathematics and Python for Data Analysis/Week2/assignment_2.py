from matplotlib import pylab
import pandas as pd
import numpy as np
import math
from scipy.linalg import solve


# Investigation function
def f(x):
    return math.sin(x / 5) * math.exp(x / 10) + 5 * math.exp(-x / 2)


# Show function graph [1, 15]
x = np.arange(1, 16)
y = map(f, x)
pylab.plot(x, list(y))
pylab.show()

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
w = solve(x, b)

# show it on graph
f_pol = lambda x: w[0] + w[1] * x
x = np.arange(16)
y = map(f_pol, x)
pylab.plot(x, list(y))
pylab.show()

# show together : Investigation function and polynomial
x = np.arange(16)
y = map(f_pol, x)
pylab.plot(x, list(y))
y = map(f, x)
pylab.plot(x, list(y))
pylab.show()