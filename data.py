import numpy as np
from scipy.integrate import odeint


def load_data(g = 9.81, h = 0.10, b = 0.25, n=3000):

    def z_derivatives(x, t):
        return [x[1], -(1/x[0])*(x[1]**2 + b*x[1] + g*x[0] - g*h)]

    time = np.linspace(0, 3, n)

    position, velocity = odeint(z_derivatives, [2e-3, 0], time).T   
    return time, position, velocity