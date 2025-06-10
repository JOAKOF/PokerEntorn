# deck.py
import random
import numpy as np

class Deck:
    """Mòdul per gestionar la baralla de 52 cartes."""
    ranks = "23456789TJQKA"
    suits = "CDHS"  # Trèvols, Diamants, Cors, Piques

    def __init__(self):
        self.cards = [r + s for r in self.ranks for s in self.suits]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, count=1):
        drawn = self.cards[:count]
        self.cards = self.cards[count:]
        return drawn
    def seed(self, seed: int = None):
        # reinicialitzar els generadors aleatoris
        random.seed(seed)
        np.random.seed(seed)
        # si la teva Deck o Tournament tenen els seus propis RNGs, inicialitza'ls també
        # p.ex. self.deck.seed(seed) o self.tournament.seed(seed)
        return
