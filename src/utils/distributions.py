import random
class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand  # Véletlenszám generátor
        self.loc = loc    # Várható érték (location)
        self.scale = scale  # Szórás (scale)

class UniformDistribution:
    pass

class CauchyDistribution:
    pass

import random

import random
import math


class LogisticDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def generate_random(self):
        u = self.rand.random()
        return self.loc + self.scale * (1.0 / (1.0 - u) - 1.0)


    def pdf(self, x):
        exponent = math.exp(-(x - self.loc) / self.scale)
        pdf_value = exponent / (self.scale * (1 + exponent)**2)
        return pdf_value



    def cdf(self, x):
        exponent = math.exp(-(x - self.loc) / self.scale)
        cdf_value = 1.0 / (1.0 + exponent)
        return cdf_value

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Probability p must be in the range [0, 1]")
        if p == 0:
            return float('-inf')
        if p == 1:
            return float('inf')
        ppf_value = self.loc - self.scale * math.log(1.0 / p - 1.0)
        return ppf_value


    def gen_rand(self):
        return self.generate_random()



    def mean(self):
        if self.scale != 0:
            return self.loc
        else:
            raise Exception("Moment undefined")

    def variance(self):
        if self.scale != 0:
            return (math.pi**2 * self.scale**2) / 3
        else:
            raise Exception("Moment undefined")

    def skewness(self):
        if self.scale != 0:
            return 0
        else:
            raise Exception("Moment undefined")

    def ex_kurtosis(self):
        if self.scale != 0:
            return 1.2
        else:
            raise Exception("Moment undefined")


    def mvsk(self):
        if self.scale != 0:
            first_moment = self.loc
            second_central_moment = (math.pi**2 * self.scale**2) / 3
            third_central_moment = 0
            excess_kurtosis = 1.2
            return [first_moment, second_central_moment, third_central_moment, excess_kurtosis]
        else:
            raise Exception("Moment undefined")








class ChiSquaredDistribution:
    def __init__(self, rand, dof):
        self.rand = rand
        self.dof = dof


import scipy.stats as stats
import scipy.stats as stats
import numpy as np


class ChiSquaredDistribution:
    def __init__(self, rand_gen, dof):
        self.rand_gen = rand_gen
        self.dof = dof


    def pdf(self, x):
        return self.rand.pdf(x, df=self.dof)

    def cdf(self, x):
        return self.rand_gen.cdf(x, df=self.dof)

    def ppf(self, p):
        return self.rand_gen.ppf(p, df=self.dof)


    def gen_rand(self):
        return self.rand_gen.logistic(loc=0, scale=1, size=None)


    def mean(self):
        mean = self.rand_gen.mean(df=self.dof)
        if np.isnan(mean):
            raise Exception("Moment undefined")
        return mean




    def variance(self):
        variance = self.rand_gen.var(df=self.dof)
        if np.isnan(variance):
            raise Exception("Moment undefined")
        return variance

    def skewness(self):
        third_moment = self.rand_gen.moment(3, df=self.dof)
        second_moment = self.rand_gen.moment(2, df=self.dof)

        if third_moment and second_moment:
            skewness = (third_moment - 3 * second_moment * self.mean() + 2 * self.mean() ** 3) / (second_moment ** 1.5)
            if not np.isnan(skewness):
                return skewness
        raise Exception("Moment undefined")









