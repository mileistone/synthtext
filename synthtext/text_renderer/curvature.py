import numpy as np

from synthtext.config import load_cfg


class Curvature(object):

    curve = lambda this, a: lambda x: a * x * x
    differential = lambda this, a: lambda x: 2 * a * x

    def __init__(self):
        load_cfg(self)

    def sample_curvature(self):
        """
        Returns the functions for the curve and differential for a and b
        """
        sgn = 1.0
        if np.random.rand() < self.p_sgn:
            sgn = -1

        a = self.a[1] * np.random.randn() + sgn * self.a[0]
        return {
            'curve': self.curve(a),
            'diff': self.differential(a),
        }
