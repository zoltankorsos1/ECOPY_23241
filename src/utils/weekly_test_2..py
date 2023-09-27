import random

import random

class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        """
        LaplaceDistribution osztály inicializálása.

        Args:
            rand (random.Random): Véletlenszám generátor objektum.
            loc (float): Laplace eloszlás várható értéke.
            scale (float): Laplace eloszlás varianciája.
        """
        self.rand = rand
        self.loc = loc
        self.scale = scale

import math

class LaplaceDistribution:
    def __init__(self, loc, scale, asymmetry):
        """
        Laplace eloszlást inicializál.

        Args:
        loc: Várható érték (valós szám).
        scale: Skála paraméter (pozitív valós szám).
        asymmetry: Aszimmetria paraméter (valós szám).
        """
        if scale <= 0:
            raise ValueError("Scale parameter must be greater than 0.")

        self.loc = loc
        self.scale = scale
        self.asymmetry = asymmetry

    def pdf(self, x):
        """
        Laplace eloszlás valószínűségi sűrűségfüggvény számítása adott x-re.

        Args:
        x: Valós szám, amelyre a valószínűségi sűrűségfüggvényt számoljuk.

        Returns:
        float: A Laplace eloszlás valószínűségi sűrűségfüggvény értéke az adott x-re.
        """
        exponent = -abs(x - self.loc) / self.scale
        pdf_value = (1 / (2 * self.scale)) * math.exp(exponent)
        return pdf_value



class LaplaceDistribution:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("Scale must be greater than 0.")

        abs_diff = abs(x - self.loc)
        return 0.5 * math.exp(-abs_diff / self.scale) / self.scale

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("Scale must be greater than 0.")

        if x < self.loc:
            return 0.5 * math.exp((x - self.loc) / self.scale)
        else:
            return 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)




import math


class LaplaceDistribution:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("Scale must be greater than 0.")

        abs_diff = abs(x - self.loc)
        return 0.5 * math.exp(-abs_diff / self.scale) / self.scale

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("Scale must be greater than 0.")

        if x < self.loc:
            return 0.5 * math.exp((x - self.loc) / self.scale)
        else:
            return 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)

    def ppf(self, p):
        if self.scale <= 0 or p < 0 or p > 1:
            raise ValueError("Scale must be greater than 0, and p must be in [0, 1] range.")

        if p == 0:
            return float('-inf')
        elif p == 1:
            return float('inf')
        else:
            if p < 0.5:
                return self.loc + self.scale * math.log(2 * p)
            else:
                return self.loc - self.scale * math.log(2 - 2 * p)






import random
import math

class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        """
        LaplaceDistribution osztály inicializálása.

        Args:
            rand (random.Random): Véletlenszám generátor objektum.
            loc (float): Laplace eloszlás várható értéke.
            scale (float): Laplace eloszlás varianciája.
        """
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def generate_sample(self):
        """
        Generál egy mintát a Laplace eloszlásból és visszaadja azt.

        Returns:
            float: Laplace eloszlásból generált minta.
        """
        # A Laplace eloszlás generálása
        u = self.rand.random()
        if u < 0.5:
            return self.loc + self.scale * self.rand.expovariate(1)
        else:
            return self.loc - self.scale * self.rand.expovariate(1)

    def pdf(self, x):
        """
        Aszimmetrikus Laplace eloszlás valószínűségi sűrűségfüggvény számítása az adott x-re.

        Args:
            x (float): Az érték, amire a valószínűségi sűrűségfüggvényt számoljuk.

        Returns:
            float: Az aszimmetrikus Laplace eloszlás valószínűségi sűrűségfüggvény értéke az x-re.
        """
        abs_diff = abs(x - self.loc)
        if x < self.loc:
            return (1 / (2 * self.scale)) * math.exp(-abs_diff / self.scale)
        else:
            return (1 / (2 * self.scale)) * math.exp(-abs_diff / self.scale)

    def gen_random(self):
        """
        Aszimmetrikus Laplace eloszlású véletlen szám generálása.

        Returns:
            float: Aszimmetrikus Laplace eloszlású véletlen szám.
        """
        return self.generate_sample()




import math


class LaplaceDistribution:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("Scale must be greater than 0.")

        abs_diff = abs(x - self.loc)
        return 0.5 * math.exp(-abs_diff / self.scale) / self.scale

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("Scale must be greater than 0.")

        if x < self.loc:
            return 0.5 * math.exp((x - self.loc) / self.scale)
        else:
            return 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)

    def ppf(self, p):
        if self.scale <= 0 or p < 0 or p > 1:
            raise ValueError("Scale must be greater than 0, and p must be in [0, 1] range.")

        if p == 0:
            return float('-inf')
        elif p == 1:
            return float('inf')
        else:
            if p < 0.5:
                return self.loc + self.scale * math.log(2 * p)
            else:
                return self.loc - self.scale * math.log(2 - 2 * p)

    def mean(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")
        return self.loc





import random
import math

class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        """
        LaplaceDistribution osztály inicializálása.

        Args:
            rand (random.Random): Véletlenszám generátor objektum.
            loc (float): Laplace eloszlás várható értéke.
            scale (float): Laplace eloszlás varianciája.
        """
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def generate_sample(self):
        """
        Generál egy mintát a Laplace eloszlásból és visszaadja azt.

        Returns:
            float: Laplace eloszlásból generált minta.
        """
        # A Laplace eloszlás generálása
        u = self.rand.random()
        if u < 0.5:
            return self.loc + self.scale * self.rand.expovariate(1)
        else:
            return self.loc - self.scale * self.rand.expovariate(1)

    def pdf(self, x):
        """
        Aszimmetrikus Laplace eloszlás valószínűségi sűrűségfüggvény számítása az adott x-re.

        Args:
            x (float): Az érték, amire a valószínűségi sűrűségfüggvényt számoljuk.

        Returns:
            float: Az aszimmetrikus Laplace eloszlás valószínűségi sűrűségfüggvény értéke az x-re.
        """
        abs_diff = abs(x - self.loc)
        if x < self.loc:
            return (1 / (2 * self.scale)) * math.exp(-abs_diff / self.scale)
        else:
            return (1 / (2 * self.scale)) * math.exp(-abs_diff / self.scale)

    def gen_random(self):
        """
        Aszimmetrikus Laplace eloszlású véletlen szám generálása.

        Returns:
            float: Aszimmetrikus Laplace eloszlású véletlen szám.
        """
        return self.generate_sample()

    def variance(self):
        """
        Eloszlás függvény varianciájának kiszámítása.

        Returns:
            float: Eloszlás függvény varianciája.
        """
        if self.scale >= 2:
            return 2 * self.scale * self.scale
        else:
            raise Exception("Moment undefined")





import math


class LaplaceDistribution:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("Scale must be greater than 0.")

        abs_diff = abs(x - self.loc)
        return 0.5 * math.exp(-abs_diff / self.scale) / self.scale

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("Scale must be greater than 0.")

        if x < self.loc:
            return 0.5 * math.exp((x - self.loc) / self.scale)
        else:
            return 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)

    def ppf(self, p):
        if self.scale <= 0 or p < 0 or p > 1:
            raise ValueError("Scale must be greater than 0, and p must be in [0, 1] range.")

        if p == 0:
            return float('-inf')
        elif p == 1:
            return float('inf')
        else:
            if p < 0.5:
                return self.loc + self.scale * math.log(2 * p)
            else:
                return self.loc - self.scale * math.log(2 - 2 * p)

    def mean(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")
        return self.loc

    def skewness(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")
        return 0.0





import random
import math

class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        """
        LaplaceDistribution osztály inicializálása.

        Args:
            rand (random.Random): Véletlenszám generátor objektum.
            loc (float): Laplace eloszlás várható értéke.
            scale (float): Laplace eloszlás varianciája.
        """
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def generate_sample(self):
        """
        Generál egy mintát a Laplace eloszlásból és visszaadja azt.

        Returns:
            float: Laplace eloszlásból generált minta.
        """
        # A Laplace eloszlás generálása
        u = self.rand.random()
        if u < 0.5:
            return self.loc + self.scale * self.rand.expovariate(1)
        else:
            return self.loc - self.scale * self.rand.expovariate(1)

    def pdf(self, x):
        """
        Aszimmetrikus Laplace eloszlás valószínűségi sűrűségfüggvény számítása az adott x-re.

        Args:
            x (float): Az érték, amire a valószínűségi sűrűségfüggvényt számoljuk.

        Returns:
            float: Az aszimmetrikus Laplace eloszlás valószínűségi sűrűségfüggvény értéke az x-re.
        """
        abs_diff = abs(x - self.loc)
        if x < self.loc:
            return (1 / (2 * self.scale)) * math.exp(-abs_diff / self.scale)
        else:
            return (1 / (2 * self.scale)) * math.exp(-abs_diff / self.scale)

    def gen_random(self):
        """
        Aszimmetrikus Laplace eloszlású véletlen szám generálása.

        Returns:
            float: Aszimmetrikus Laplace eloszlású véletlen szám.
        """
        return self.generate_sample()

    def variance(self):
        """
        Eloszlás függvény varianciájának kiszámítása.

        Returns:
            float: Eloszlás függvény varianciája.
        """
        if self.scale >= 2:
            return 2 * self.scale * self.scale
        else:
            raise Exception("Moment undefined")

    def ex_kurtosis(self):
        """
        Eloszlás függvény többlet csúcsosságának (excess kurtosis) kiszámítása.

        Returns:
            float: Eloszlás függvény többlet csúcsossága.
        """
        if self.scale >= 2:
            return 12 / (self.scale * self.scale)
        else:
            raise Exception("Moment undefined")





import math


class LaplaceDistribution:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("Scale must be greater than 0.")

        abs_diff = abs(x - self.loc)
        return 0.5 * math.exp(-abs_diff / self.scale) / self.scale

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("Scale must be greater than 0.")

        if x < self.loc:
            return 0.5 * math.exp((x - self.loc) / self.scale)
        else:
            return 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)

    def ppf(self, p):
        if self.scale <= 0 or p < 0 or p > 1:
            raise ValueError("Scale must be greater than 0, and p must be in [0, 1] range.")

        if p == 0:
            return float('-inf')
        elif p == 1:
            return float('inf')
        else:
            if p < 0.5:
                return self.loc + self.scale * math.log(2 * p)
            else:
                return self.loc - self.scale * math.log(2 - 2 * p)

    def mean(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")
        return self.loc

    def mvsk(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")

        mean = self.loc
        variance = 2 * (self.scale ** 2)
        skewness = 0.0
        kurtosis = 3.0

        return [mean, variance, skewness, kurtosis]













class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        """
        Pareto eloszlást inicializál.

        Args:
        rand: Véletlenszám generátor objektum.
        scale: Skála paraméter (k > 0).
        shape: Alak paraméter (α > 0).
        """
        if scale <= 0 or shape <= 0:
            raise ValueError("Scale and shape parameters must be greater than 0.")

        self.rand = rand
        self.scale = scale
        self.shape = shape

    def generate_random(self, n):
        """
        Pareto eloszlás szerinti véletlenszámokat generál.

        Args:
        n: Generálandó véletlenszámok száma.

        Returns:
        List of n véletlenszám a Pareto eloszlás szerint.
        """
        if n <= 0:
            raise ValueError("Number of random samples must be greater than 0.")

        random_samples = [self.scale * (self.rand.random() ** (-1 / self.shape)) for _ in range(n)]
        return random_samples





import random
import math

class ParetoDistribution:
    def __init__(self, rand, loc, scale):
        """
        ParetoDistribution osztály inicializálása.

        Args:
            rand (random.Random): Véletlenszám generátor objektum.
            loc (float): Pareto eloszlás várható értéke.
            scale (float): Pareto eloszlás skálája.
        """
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def generate_sample(self):
        """
        Generál egy mintát a Pareto eloszlásból és visszaadja azt.

        Returns:
            float: Pareto eloszlásból generált minta.
        """
        # A Pareto eloszlás generálása
        u = self.rand.random()
        return self.loc / math.pow(1 - u, 1 / self.scale)

    def pdf(self, x):
        """
        Aszimmetrikus Laplace eloszlás eloszlásfüggvény számítása az adott x-re.

        Args:
            x (float): Az érték, amire a valószínűségi sűrűségfüggvényt számoljuk.

        Returns:
            float: Az aszimmetrikus Laplace eloszlás eloszlásfüggvény értéke az x-re.
        """
        if x >= self.loc:
            return (self.scale / self.loc) * math.pow(x / self.loc, -(self.scale + 1))
        else:
            return 0





class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        """
        Pareto eloszlást inicializál.

        Args:
        rand: Véletlenszám generátor objektum.
        scale: Skála paraméter (k > 0).
        shape: Alak paraméter (α > 0).
        """
        if scale <= 0 or shape <= 0:
            raise ValueError("Scale and shape parameters must be greater than 0.")

        self.rand = rand
        self.scale = scale
        self.shape = shape

    def generate_random(self, n):
        """
        Pareto eloszlás szerinti véletlenszámokat generál.

        Args:
        n: Generálandó véletlenszámok száma.

        Returns:
        List of n véletlenszám a Pareto eloszlás szerint.
        """
        if n <= 0:
            raise ValueError("Number of random samples must be greater than 0.")

        random_samples = [self.scale * (self.rand.random() ** (-1 / self.shape)) for _ in range(n)]
        return random_samples

    def cdf(self, x):
        """
        Pareto eloszlás kumulatív eloszlásfüggvény számítása.

        Args:
        x: Érték, amelyre a kumulatív eloszlásfüggvényt számoljuk.

        Returns:
        float: A Pareto eloszlás kumulatív eloszlásfüggvényének értéke az adott x-re.
        """
        if x < self.scale:
            return 0.0
        else:
            return 1.0 - (self.scale / x) ** self.shape





class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        """
        Pareto eloszlást inicializál.

        Args:
        rand: Véletlenszám generátor objektum.
        scale: Skála paraméter (k > 0).
        shape: Alak paraméter (α > 0).
        """
        if scale <= 0 or shape <= 0:
            raise ValueError("Scale and shape parameters must be greater than 0.")

        self.rand = rand
        self.scale = scale
        self.shape = shape

    def generate_random(self, n):
        """
        Pareto eloszlás szerinti véletlenszámokat generál.

        Args:
        n: Generálandó véletlenszámok száma.

        Returns:
        List of n véletlenszám a Pareto eloszlás szerint.
        """
        if n <= 0:
            raise ValueError("Number of random samples must be greater than 0.")

        random_samples = [self.scale * (self.rand.random() ** (-1 / self.shape)) for _ in range(n)]
        return random_samples

    def cdf(self, x):
        """
        Pareto eloszlás kumulatív eloszlásfüggvény számítása.

        Args:
        x: Érték, amelyre a kumulatív eloszlásfüggvényt számoljuk.

        Returns:
        float: A Pareto eloszlás kumulatív eloszlásfüggvényének értéke az adott x-re.
        """
        if x < self.scale:
            return 0.0
        else:
            return 1.0 - (self.scale / x) ** self.shape

    def ppf(self, p):
        """
        Pareto eloszlás inverz kumulatív eloszlásfüggvény számítása.

        Args:
        p: Valószínűség érték [0, 1] tartományban.

        Returns:
        float: Az inverz kumulatív eloszlásfüggvény értéke az adott valószínűségi értékre.
        """
        if p < 0 or p > 1:
            raise ValueError("Probability must be in [0, 1] range.")

        if p == 0:
            return self.scale
        elif p == 1:
            return float('inf')
        else:
            return self.scale / ((1 - p) ** (1 / self.shape))






import random
import math


class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        """
        Pareto eloszlást inicializál.

        Args:
        rand: Véletlenszám generátor objektum.
        scale: Skála paraméter (k > 0).
        shape: Alak paraméter (α > 0).
        """
        if scale <= 0 or shape <= 0:
            raise ValueError("Scale and shape parameters must be greater than 0.")

        self.rand = rand
        self.scale = scale
        self.shape = shape

    def generate_random(self, n):
        """
        Pareto eloszlás szerinti véletlenszámokat generál.

        Args:
        n: Generálandó véletlenszámok száma.

        Returns:
        List of n véletlenszám a Pareto eloszlás szerint.
        """
        if n <= 0:
            raise ValueError("Number of random samples must be greater than 0.")

        random_samples = [self.scale * (self.rand.random() ** (-1 / self.shape)) for _ in range(n)]
        return random_samples

    def cdf(self, x):
        """
        Pareto eloszlás kumulatív eloszlásfüggvény számítása.

        Args:
        x: Érték, amelyre a kumulatív eloszlásfüggvényt számoljuk.

        Returns:
        float: A Pareto eloszlás kumulatív eloszlásfüggvényének értéke az adott x-re.
        """
        if x < self.scale:
            return 0.0
        else:
            return 1.0 - (self.scale / x) ** self.shape

    def ppf(self, p):
        """
        Pareto eloszlás inverz kumulatív eloszlásfüggvény számítása.

        Args:
        p: Valószínűség érték [0, 1] tartományban.

        Returns:
        float: Az inverz kumulatív eloszlásfüggvény értéke az adott valószínűségi értékhez.
        """
        if p < 0 or p > 1:
            raise ValueError("Probability must be in [0, 1] range.")

        if p == 0:
            return self.scale
        elif p == 1:
            return float('inf')
        else:
            return self.scale / ((1 - p) ** (1 / self.shape))

    def gen_random(self):
        """
        Aszimmetrikus Laplace-eloszlású véletlen szám generálása.

        Returns:
        float: Aszimmetrikus Laplace-eloszlású véletlen szám.
        """
        u = self.rand.random()
        if u < 0.5:
            return self.scale * (1 - 2 * u) ** (-1 / self.shape)
        else:
            return self.scale * (2 * u - 1) ** (-1 / self.shape)





import random
import math


class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        """
        Pareto eloszlást inicializál.

        Args:
        rand: Véletlenszám generátor objektum.
        scale: Skála paraméter (k > 0).
        shape: Alak paraméter (α > 0).
        """
        if scale <= 0 or shape <= 0:
            raise ValueError("Scale and shape parameters must be greater than 0.")

        self.rand = rand
        self.scale = scale
        self.shape = shape

    def generate_random(self, n):
        """
        Pareto eloszlás szerinti véletlenszámokat generál.

        Args:
        n: Generálandó véletlenszámok száma.

        Returns:
        List of n véletlenszám a Pareto eloszlás szerint.
        """
        if n <= 0:
            raise ValueError("Number of random samples must be greater than 0.")

        random_samples = [self.scale * (self.rand.random() ** (-1 / self.shape)) for _ in range(n)]
        return random_samples

    def cdf(self, x):
        """
        Pareto eloszlás kumulatív eloszlásfüggvény számítása.

        Args:
        x: Érték, amelyre a kumulatív eloszlásfüggvényt számoljuk.

        Returns:
        float: A Pareto eloszlás kumulatív eloszlásfüggvényének értéke az adott x-re.
        """
        if x < self.scale:
            return 0.0
        else:
            return 1.0 - (self.scale / x) ** self.shape

    def ppf(self, p):
        """
        Pareto eloszlás inverz kumulatív eloszlásfüggvény számítása.

        Args:
        p: Valószínűség érték [0, 1] tartományban.

        Returns:
        float: Az inverz kumulatív eloszlásfüggvény értéke az adott valószínűségi értékhez.
        """
        if p < 0 or p > 1:
            raise ValueError("Probability must be in [0, 1] range.")

        if p == 0:
            return self.scale
        elif p == 1:
            return float('inf')
        else:
            return self.scale / ((1 - p) ** (1 / self.shape))

    def gen_random(self):
        """
        Aszimmetrikus Laplace-eloszlású véletlen szám generálása.

        Returns:
        float: Aszimmetrikus Laplace-eloszlású véletlen szám.
        """
        u = self.rand.random()
        if u < 0.5:
            return self.scale * (1 - 2 * u) ** (-1 / self.shape)
        else:
            return self.scale * (2 * u - 1) ** (-1 / self.shape)

    def mean(self):
        """
        Pareto eloszlásfüggvény várható érték számítása.

        Returns:
        float: A Pareto eloszlásfüggvény várható értéke.
        """
        if self.shape <= 1:
            raise Exception("Moment undefined")
        return (self.shape * self.scale) / (self.shape - 1)




import random
import math


class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        """
        Pareto eloszlást inicializál.

        Args:
        rand: Véletlenszám generátor objektum.
        scale: Skála paraméter (k > 0).
        shape: Alak paraméter (α > 0).
        """
        if scale <= 0 or shape <= 0:
            raise ValueError("Scale and shape parameters must be greater than 0.")

        self.rand = rand
        self.scale = scale
        self.shape = shape

    def generate_random(self, n):
        """
        Pareto eloszlás szerinti véletlenszámokat generál.

        Args:
        n: Generálandó véletlenszámok száma.

        Returns:
        List of n véletlenszám a Pareto eloszlás szerint.
        """
        if n <= 0:
            raise ValueError("Number of random samples must be greater than 0.")

        random_samples = [self.scale * (self.rand.random() ** (-1 / self.shape)) for _ in range(n)]
        return random_samples

    def cdf(self, x):
        """
        Pareto eloszlás kumulatív eloszlásfüggvény számítása.

        Args:
        x: Érték, amelyre a kumulatív eloszlásfüggvényt számoljuk.

        Returns:
        float: A Pareto eloszlás kumulatív eloszlásfüggvényének értéke az adott x-re.
        """
        if x < self.scale:
            return 0.0
        else:
            return 1.0 - (self.scale / x) ** self.shape

    def ppf(self, p):
        """
        Pareto eloszlás inverz kumulatív eloszlásfüggvény számítása.

        Args:
        p: Valószínűség érték [0, 1] tartományban.

        Returns:
        float: Az inverz kumulatív eloszlásfüggvény értéke az adott valószínűségi értékhez.
        """
        if p < 0 or p > 1:
            raise ValueError("Probability must be in [0, 1] range.")

        if p == 0:
            return self.scale
        elif p == 1:
            return float('inf')
        else:
            return self.scale / ((1 - p) ** (1 / self.shape))

    def gen_random(self):
        """
        Aszimmetrikus Laplace-eloszlású véletlen szám generálása.

        Returns:
        float: Aszimmetrikus Laplace-eloszlású véletlen szám.
        """
        u = self.rand.random()
        if u < 0.5:
            return self.scale * (1 - 2 * u) ** (-1 / self.shape)
        else:
            return self.scale * (2 * u - 1) ** (-1 / self.shape)

    def mean(self):
        """
        Pareto eloszlásfüggvény várható érték számítása.

        Returns:
        float: A Pareto eloszlásfüggvény várható értéke.
        """
        if self.shape <= 1:
            raise Exception("Moment undefined")
        return (self.shape * self.scale) / (self.shape - 1)

    def variance(self):
        """
        Pareto eloszlásfüggvény variancia (szórásnégyzet) számítása.

        Returns:
        float: A Pareto eloszlásfüggvény varianciája (szórásnégyzete).
        """
        if self.shape <= 2:
            raise Exception("Moment undefined")
        return (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))






import random
import math


class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        """
        Pareto eloszlást inicializál.

        Args:
        rand: Véletlenszám generátor objektum.
        scale: Skála paraméter (k > 0).
        shape: Alak paraméter (α > 0).
        """
        if scale <= 0 or shape <= 0:
            raise ValueError("Scale and shape parameters must be greater than 0.")

        self.rand = rand
        self.scale = scale
        self.shape = shape

    def generate_random(self, n):
        """
        Pareto eloszlás szerinti véletlenszámokat generál.

        Args:
        n: Generálandó véletlenszámok száma.

        Returns:
        List of n véletlenszám a Pareto eloszlás szerint.
        """
        if n <= 0:
            raise ValueError("Number of random samples must be greater than 0.")

        random_samples = [self.scale * (self.rand.random() ** (-1 / self.shape)) for _ in range(n)]
        return random_samples

    def cdf(self, x):
        """
        Pareto eloszlás kumulatív eloszlásfüggvény számítása.

        Args:
        x: Érték, amelyre a kumulatív eloszlásfüggvényt számoljuk.

        Returns:
        float: A Pareto eloszlás kumulatív eloszlásfüggvényének értéke az adott x-re.
        """
        if x < self.scale:
            return 0.0
        else:
            return 1.0 - (self.scale / x) ** self.shape

    def ppf(self, p):
        """
        Pareto eloszlás inverz kumulatív eloszlásfüggvény számítása.

        Args:
        p: Valószínűség érték [0, 1] tartományban.

        Returns:
        float: Az inverz kumulatív eloszlásfüggvény értéke az adott valószínűségi értékhez.
        """
        if p < 0 or p > 1:
            raise ValueError("Probability must be in [0, 1] range.")

        if p == 0:
            return self.scale
        elif p == 1:
            return float('inf')
        else:
            return self.scale / ((1 - p) ** (1 / self.shape))

    def gen_random(self):
        """
        Aszimmetrikus Laplace-eloszlású véletlen szám generálása.

        Returns:
        float: Aszimmetrikus Laplace-eloszlású véletlen szám.
        """
        u = self.rand.random()
        if u < 0.5:
            return self.scale * (1 - 2 * u) ** (-1 / self.shape)
        else:
            return self.scale * (2 * u - 1) ** (-1 / self.shape)

    def mean(self):
        """
        Pareto eloszlásfüggvény várható érték számítása.

        Returns:
        float: A Pareto eloszlásfüggvény várható értéke.
        """
        if self.shape <= 1:
            raise Exception("Moment undefined")
        return (self.shape * self.scale) / (self.shape - 1)

    def variance(self):
        """
        Pareto eloszlásfüggvény variancia (szórásnégyzet) számítása.

        Returns:
        float: A Pareto eloszlásfüggvény varianciája (szórásnégyzete).
        """
        if self.shape <= 2:
            raise Exception("Moment undefined")
        return (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))

    def skewness(self):
        """
        Pareto eloszlásfüggvény ferdeség számítása.

        Returns:
        float: A Pareto eloszlásfüggvény ferdesége.
        """
        if self.shape <= 3:
            raise Exception("Moment undefined")
        return (2 * (1 + self.shape)) / ((self.shape - 3) * math.sqrt((self.shape - 2) / self.shape))






import random
import math


class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        """
        Pareto eloszlást inicializál.

        Args:
        rand: Véletlenszám generátor objektum.
        scale: Skála paraméter (k > 0).
        shape: Alak paraméter (α > 0).
        """
        if scale <= 0 or shape <= 0:
            raise ValueError("Scale and shape parameters must be greater than 0.")

        self.rand = rand
        self.scale = scale
        self.shape = shape

    def generate_random(self, n):
        """
        Pareto eloszlás szerinti véletlenszámokat generál.

        Args:
        n: Generálandó véletlenszámok száma.

        Returns:
        List of n véletlenszám a Pareto eloszlás szerint.
        """
        if n <= 0:
            raise ValueError("Number of random samples must be greater than 0.")

        random_samples = [self.scale * (self.rand.random() ** (-1 / self.shape)) for _ in range(n)]
        return random_samples

    def cdf(self, x):
        """
        Pareto eloszlás kumulatív eloszlásfüggvény számítása.

        Args:
        x: Érték, amelyre a kumulatív eloszlásfüggvényt számoljuk.

        Returns:
        float: A Pareto eloszlás kumulatív eloszlásfüggvényének értéke az adott x-re.
        """
        if x < self.scale:
            return 0.0
        else:
            return 1.0 - (self.scale / x) ** self.shape

    def ppf(self, p):
        """
        Pareto eloszlás inverz kumulatív eloszlásfüggvény számítása.

        Args:
        p: Valószínűség érték [0, 1] tartományban.

        Returns:
        float: Az inverz kumulatív eloszlásfüggvény értéke az adott valószínűségi értékhez.
        """
        if p < 0 or p > 1:
            raise ValueError("Probability must be in [0, 1] range.")

        if p == 0:
            return self.scale
        elif p == 1:
            return float('inf')
        else:
            return self.scale / ((1 - p) ** (1 / self.shape))

    def gen_random(self):
        """
        Aszimmetrikus Laplace-eloszlású véletlen szám generálása.

        Returns:
        float: Aszimmetrikus Laplace-eloszlású véletlen szám.
        """
        u = self.rand.random()
        if u < 0.5:
            return self.scale * (1 - 2 * u) ** (-1 / self.shape)
        else:
            return self.scale * (2 * u - 1) ** (-1 / self.shape)

    def mean(self):
        """
        Pareto eloszlásfüggvény várható érték számítása.

        Returns:
        float: A Pareto eloszlásfüggvény várható értéke.
        """
        if self.shape <= 1:
            raise Exception("Moment undefined")
        return (self.shape * self.scale) / (self.shape - 1)

    def variance(self):
        """
        Pareto eloszlásfüggvény variancia (szórásnégyzet) számítása.

        Returns:
        float: A Pareto eloszlásfüggvény varianciája (szórásnégyzete).
        """
        if self.shape <= 2:
            raise Exception("Moment undefined")
        return (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))

    def skewness(self):
        """
        Pareto eloszlásfüggvény ferdeség számítása.

        Returns:
        float: A Pareto eloszlásfüggvény ferdesége.
        """
        if self.shape <= 3:
            raise Exception("Moment undefined")
        return (2 * (1 + self.shape)) / ((self.shape - 3) * math.sqrt((self.shape - 2) / self.shape))

    def ex_kurtosis(self):
        """
        Pareto eloszlásfüggvény többlet csúcsosság számítása.

        Returns:
        float: A Pareto eloszlásfüggvény többlet csúcsossága.
        """
        if self.shape <= 3:
            raise Exception("Moment undefined")
        return 6 / (self.shape - 3)







import random
import math

class ParetoDistribution:
    def __init__(self, rand, loc, scale):
        """
        ParetoDistribution osztály inicializálása.

        Args:
            rand (random.Random): Véletlenszám generátor objektum.
            loc (float): Pareto eloszlás várható értéke.
            scale (float): Pareto eloszlás skálája.
        """
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def generate_sample(self):
        """
        Generál egy mintát a Pareto eloszlásból és visszaadja azt.

        Returns:
            float: Pareto eloszlásból generált minta.
        """
        # A Pareto eloszlás generálása
        u = self.rand.random()
        return self.loc / math.pow(1 - u, 1 / self.scale)

    def pdf(self, x):
        """
        Aszimmetrikus Laplace eloszlás eloszlásfüggvény számítása az adott x-re.

        Args:
            x (float): Az érték, amire a valószínűségi sűrűségfüggvényt számoljuk.

        Returns:
            float: Az aszimmetrikus Laplace eloszlás eloszlásfüggvény értéke az x-re.
        """
        if x >= self.loc:
            return (self.scale / self.loc) * math.pow(x / self.loc, -(self.scale + 1))
        else:
            return 0

    def ppf(self, p):
        """
        Eloszlás inverz kumulatív eloszlásfüggvényének számítása.

        Args:
            p (float): A valószínűség, amire az inverz kumulatív eloszlásfüggvényt számoljuk.

        Returns:
            float: Az inverz kumulatív eloszlásfüggvény értéke a p valószínűséghez.
        """
        if 0 < p < 1:
            return self.loc / math.pow(1 - p, 1 / self.scale)
        else:
            raise ValueError("p must be in the range (0, 1)")





