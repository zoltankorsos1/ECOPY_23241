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
    def __init__(self, rand, location, scale):
        self.rand = rand
        self.location = location
        self.scale = scale

    def generate_random(self):
        import math
        u = self.rand.random()  # Véletlenszerű szám az [0, 1) tartományból
        x = self.location + self.scale * math.log(u / (1 - u))  # Logistic eloszlás generálása
        return x




    def pdf(self, x):
        exponent = -(x - self.location) / self.scale
        pdf_value =math.exp(exponent) / (self.scale * (1 + math.exp(exponent))**2)
        return pdf_value


    def cdf(self, x):
        exponent = math.exp(-(x - self.location) / self.scale)
        cdf_value = 1 / (1 + exponent)
        return cdf_value


    def ppf(self, p):
        import math
        if p < 0 or p > 1:
            raise ValueError("A valószínűségi értéknek az [0, 1] tartományban kell lennie.")
        if p == 0:
            return float("-inf")
        if p == 1:
            return float("inf")
        ppf_value = self.location + self.scale * math.log(p / (1 - p))
        return ppf_value

    def gen_rand(self):
        return self.generate_random()


    def mean(self):
        if self.scale != 0:
            return self.location
        else:
            raise Exception("Moment undefined")

    def variance(self):
        if self.scale != 0:
            return (math.pi**2 * self.scale**2) / 3
        else:
            raise Exception("Moment undefined")


    def skewness(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        skewness = 0
        return skewness

    def ex_kurtosis(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        ex_kurtosis = 1.2
        return ex_kurtosis


    def mvsk(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]




class ChiSquaredDistribution:
    def __init__(self, rand, dof):
        self.rand = rand
        self.dof = dof

    def pdf(self, x):
        import math
        if x < 0:
            return 0
        coefficient = (1 / (2**(self.dof / 2))) * (1 / math.gamma(self.dof / 2))
        exponent = (-x / 2)
        return coefficient * (x**(self.dof / 2 - 1)) * math.exp(exponent)

    def cdf(self, x):
        import scipy.special
        if x < 0:
            return 0
        return scipy.special.gammainc(self.dof / 2, x / 2)


    def ppf(self, p):
        import scipy.special
        if p < 0 or p > 1:
            raise ValueError("Moment undefined")
        return 2 * scipy.special.gammaincinv(self.dof / 2, p)

    def gen_rand(self):
        import random
        u = random.uniform(0, 1)
        return self.ppf(u)
    def mean(self):
        try: return self.dof
        except: raise ValueError("Moment undefined")

    def variance(self):
        try: return 2 * self.dof
        except:
            raise ValueError("Moment undefined")


    def skewness(self):
        import math
        try: return math.sqrt(8 / self.dof)
        except:
            raise ValueError("Moment undefined")
    def ex_kurtosis(self):
        try: return 12 / self.dof
        except:
            raise ValueError("Moment undefined")


    def mvsk(self):
            mean = self.mean()
            variance = self.variance()
            skewness = self.skewness()
            ex_kurtosis = self.ex_kurtosis()
            try: return [mean, variance, skewness, ex_kurtosis]
            except:
                raise ValueError("Moment undefined")













