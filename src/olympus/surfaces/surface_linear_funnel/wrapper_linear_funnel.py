#!/usr/bin/env python

from itertools import product

import numpy as np

from olympus import Logger
from olympus.surfaces import AbstractSurface


class LinearFunnel(AbstractSurface):
    def __init__(self, param_dim=2, noise=None):
        """Linear funnel function.

        Args:
            param_dim (int): Number of input dimensions. Default is 2.
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        value_dim = 1
        task = 'regression'
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        message = "LinearFunnel has an infinite number of minima at 0.45 < x_i < 0.55, for each x_i in x"
        Logger.log(message, "INFO")
        # minimum at the centre
        params = [0.5] * self.param_dim
        value = self._run(params)
        return [{"params": params, "value": value}]

    @property
    def maxima(self):
        message = "LinearFunnel has an infinite number of maxima"
        Logger.log(message, "INFO")
        # some maxima
        maxima = []
        params = product([0, 1], repeat=self.param_dim)
        for param in params:
            param = list(param)
            value = self._run(param)
            maxima.append({"params": param, "value": value})
        return maxima

    def _run(self, params):
        params = np.array(params)
        params = 10 * params - 5  # rescale onto [-5, 5]
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
        result = 5
        for bound in bounds[::-1]:
            if np.amax(np.abs(params)) < bound:
                result -= 1
        result = np.amin([4, result])

        if self.noise is None:
            return result
        else:
            return self.noise(result)
