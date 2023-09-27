import random

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand  # Véletlenszám generátor
        self.loc = loc    # Várható érték (location)
        self.scale = scale  # Szórás (scale)