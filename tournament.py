# tournament.py
class Tournament:
    """
    Gestiona aspectes del torneig:
      - Assigna stacks inicials.
      - Rota el botó i les blinds.
      - Permet actualitzar els stacks.
    """


    def __init__(self, agents, initial_stack=1000, small_blind=10, big_blind=20, actual_stacks = {}):
        self.agents = agents
        if actual_stacks != {}:
            self.stacks = actual_stacks.copy()
        else:
            self.stacks = {agent: initial_stack for agent in agents}
        self.dealer_idx = 0
        self.small_blind = small_blind
        self.big_blind = big_blind

    def assign_blinds(self):
        n = len(self.agents)
        dealer = self.agents[self.dealer_idx]
        small_blind = self.agents[(self.dealer_idx + 1) % n]
        big_blind = self.agents[(self.dealer_idx + 2) % n]

        # Descompta les blinds dels stacks
        self.stacks[small_blind] -= self.small_blind
        self.stacks[big_blind] -= self.big_blind

        blinds = {dealer: 0, small_blind: self.small_blind, big_blind: self.big_blind}
        return dealer, small_blind, big_blind, blinds

    def rotate(self):
        self.dealer_idx = (self.dealer_idx + 1) % len(self.agents)

    def update_stack(self, agent, amount):
        self.stacks[agent] += amount

    def get_stack(self, agent):
        return self.stacks[agent]
    
    # elimina els que estan a 0 en tancar la mà
    def drop_busted(self):
        self.agents = [a for a in self.agents if self.stacks[a] > 0]
