#!/usr/bin/python

import scipy.integrate as integrate


class SizeDensity():
    """Size density class, initialized from a function sigma that gives the size density between a minimum and maximum size. Calling the class instance with a value x < 1 returns amax*sigma(amax*x)/rho_d.

    Args:
        sigma (callable): size density function
        size_range: size range of the size density function
    """
    def __init__(self, sigma, size_range, sigma_integral=None):
        # sigma = sigma(a) between amin and amax
        self.amin = size_range[0]
        self.amax = size_range[1]

        if callable(sigma):
            if sigma_integral is None:
                self.rhod = integrate.quad(sigma, self.amin, self.amax)[0]
            else:
                self.rhod = sigma_integral
            self.f = lambda x: self.amax*sigma(self.amax*x)/self.rhod
            self.monodisperse = False
        else:
            print('Monodisperse size distribution')
            self.rhod = sigma
            self.f = lambda x: self.amax
            self.monodisperse = True
        # Empty list of poles
        self.poles = []

    def __call__(self, x):
        """Return amax*sigma(amax*x)/rho_d"""
        return self.f(x)

    def sigma(self, x):
        return self.f(x/self.amax)/self.amax

    def integrate(self, func):
        """Integrate sigma*func over all sizes"""
        if self.monodisperse is False:
            g = lambda x: self.f(x)*func(self.amax*x)
            return integrate.fixed_quad(g, self.amin/self.amax, 1, n=50)[0]
        else:
            return func(self.amax)
