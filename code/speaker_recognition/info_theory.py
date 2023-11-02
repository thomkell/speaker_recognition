
import numpy as np
from scipy.stats import norm, uniform


class Distribution:
    """
    Class for generating samples and estimating their likelihood
    Example:
    U = Distribution(scipy.stats.chi2(3))  # Chi-squared 3 degrees of freedom
    u = U.draw((2,5))  Draw 10 samples as a 2x5 matrix
    p_u = U.likelihood(u)  Estimate the likelihood of each sample
    """

    def __init__(self, density):
        """
        :param density:  A scipy.stats density
        Example:  scipy.stats.uniform specifies a uniform density over
        the interval [a, b] as loc=a, scale=b-a
        """
        self.density = density

    def draw(self, shape):
        """
        Draw a set of samples specified by tuple shape
        :param shape: sample tensor size
        :return: samples
        """
        values = self.density.rvs(size=shape)
        return values

    def likelihood(self, samples):
        """
        Return the likelihood of samples assuming they are drawn
        from this distribution
        :param samples:
        :return: P(samples)
        """
        p = self.density.pdf(samples)
        return p
