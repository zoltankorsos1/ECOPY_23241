Test 2
Args:
            rand: Véletlenszám generátor objektum.
            loc: Laplace-eloszlás várható értéke.
            scale: Laplace-eloszlás varianciája.

        Attributes:
            rand (objektum): A véletlenszám generátor objektum.
            loc (float): A várható érték.
            scale (float): A variancia.
Laplace:
Ez a más: Ha az eloszlás függvénynek (self.scale) nincs értéke (0), akkor raise Exception ('Moment undefined')....
if self.scale == 0:
    raise Exception("Moment undefined")

6. Mean
Jo
def mean(self):
    if self.scale == 0:
        raise Exception("Moment undefined")
    else:
        return self.loc
chat
    def mean(self):

        if hasattr(self, 'loc'):
            return self.loc
        else:
            raise Exception("Moment undefined")

7. Variance
jo
def variance(self):
    if self.scale == 0:
        raise Exception("Moment undefined")
    return 2 * (self.scale ** 2)

chat
def variance(self):
    if hasattr(self, 'scale'):
        return 2 * self.scale ** 2
    else:
        raise Exception("Moment undefined")

8. Skewness
j
def skewness(self):
    if self.scale == 0:
        raise Exception("Moment undefined")
    return 0
c
def skewness(self):
    if hasattr(self, 'scale'):
        return 0
    else:
        raise Exception("Moment undefined")

9.
def ex_kurtosis(self):
    if self.scale == 0:
        raise Exception("Moment undefined")
    return 3.0

def ex_kurtosis(self):

    if hasattr(self, 'scale'):
        return 3
    else:
        raise Exception("Moment undefined")


10.

def mvsk(self):
    if self.scale == 0:
        raise Exception("Moment undefined")
    return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]

def mvsk(self):

    return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]


Pareto:
1-től 4-ig if és else megcserélve


1.
class ParetoDistribution:
def __init__(self, rand, scale, shape):
    self.rand = rand
    self.scale = scale
    self.shape = shape

import random

class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        self.rand = rand
        self.scale = scale
        self.shape = shape

    def generate_random_sample(self, size):
        # A véletlenszámok generálása Pareto eloszlás alapján
        random_sample = [self.scale * (self.rand.random() ** (-1/self.shape)) for _ in range(size)]
        return random_sample

2. PDF

    def pdf(self, x):
        if x < self.scale:
            return 0
        else:
            return self.shape * (self.scale ** self.shape) / (x ** (self.shape + 1))

    def pdf(self, x):
        if x >= self.scale:
            return (self.shape * self.scale ** self.shape) / (x ** (self.shape + 1))
        else:
            return 0.0

3. CDF
def cdf(self, x):
    if x < self.scale:
        return 0
    else:
        return 1 - (self.scale / x) ** self.shape

def cdf(self, x):
    if x >= self.scale:
        return 1 - (self.scale / x) ** self.shape
    else:
        return 0.0

4. PPF

def ppf(self, p):
    if p < 0 or p > 1:
        raise ValueError("p értékének 0 és 1 között kell lennie")
    else:
        return self.scale / ((1 - p) ** (1 / self.shape))

def ppf(self, p):
    if 0 <= p <= 1:
        return self.scale / (1 - p) ** (1 / self.shape)
    else:
        raise ValueError("Az 'p' értéke 0 és 1 között kell, hogy legyen.")

5. Gen random

def gen_rand(self):
    u = random.uniform(0, 1)
    return self.ppf(u)

    def gen_random(self):
        u = self.rand.random() - 0.5
        if u >= 0:
            return self.scale * (-math.log(1 - 2*u)) ** (1/self.shape)
        else:
            return -self.scale * (-math.log(1 + 2*u)) ** (1/self.shape)


6. MEAN
def mean(self):
    if self.shape <= 1:
        return math.inf
    else:
        return (self.shape * self.scale) / (self.shape - 1)

    def mean(self):
        if self.shape <= 1:
            raise Exception("Moment undefined")
        else:
            return (self.shape * self.scale) / (self.shape - 1)


7. Variance
def variance(self):
    if self.shape <= 2:
        return math.inf
    else:
        return (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))

    def variance(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")
        else:
            return 2 * (self.scale ** 2)


8. Skewness
def skewness(self):
    if self.shape <= 3:
        return math.inf
    else:
        return (2 * (1 + self.shape)) / (self.shape - 3) * math.sqrt((self.shape - 2) / self.shape)

    def skewness(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")
        else:
            return 0.0

9. Kurtosis:
def ex_kurtosis(self):
    if self.shape <= 4:
        return math.inf
    else:
        return 6 * ((self.shape ** 3 + self.shape ** 2 - 6 * self.shape - 2) / (
                self.shape * (self.shape - 3) * (self.shape - 4)))

    def ex_kurtosis(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")
        else:
            return 3.0

10. MVSK

def mvsk(self):
    return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]






