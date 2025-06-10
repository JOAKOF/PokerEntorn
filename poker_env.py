# poker_env.py
import random
import copy
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces
from deck import Deck
from tournament import Tournament
from treys import Card, Evaluator


def hand_strength(hole, board, n_sim=50):
    """
    Calcula la força de la mà actual amb una estimació entre 0 i 1.
    Utilitza treys.Evaluator on 0 = pitjor mà possible, 1 = nuts absoluts.
    """
    evaluator = Evaluator()
    
    # Primer de tot, comprovem que tenim cartes vàlides
    if not hole or len(hole) != 2:
        return 0.0
    if len(board) > 5:
        return 0.0
    
    try:
        # Si ja tenim les 5 cartes del river, podem calcular directament
        if len(board) == 5:
            hole_cards = [Card.new(c[0] + c[1].lower()) for c in hole]
            board_cards = [Card.new(c[0] + c[1].lower()) for c in board]
            score = evaluator.evaluate(hole_cards, board_cards)
            return 1.0 - score / 7462.0

        # Si encara no s'ha completat el river, fem Monte-Carlo contra mans aleatòries
        if len(board) < 5:
            wins = ties = 0
            
            for _ in range(n_sim):
                # Creem una nova baralla per cada simulació
                temp_deck = Deck()
                temp_deck.shuffle()
                
                # Traiem les cartes que ja coneixem
                known = hole + board
                temp_deck.cards = [c for c in temp_deck.cards if c not in known]
                
                # Assegurem-nos que tenim prou cartes per simular
                cards_needed = (5 - len(board)) + 2  # completar board + oponent
                if len(temp_deck.cards) < cards_needed:
                    continue
                
                # Completem el board amb cartes aleatòries
                sim_board = board + temp_deck.draw(5 - len(board))
                opp_hole = temp_deck.draw(2)
                
                # Avaluem ambdues mans
                hole_cards = [Card.new(c[0] + c[1].lower()) for c in hole]
                board_cards = [Card.new(c[0] + c[1].lower()) for c in sim_board]
                opp_hole_cards = [Card.new(c[0] + c[1].lower()) for c in opp_hole]
                
                my_score = evaluator.evaluate(hole_cards, board_cards)
                opp_score = evaluator.evaluate(opp_hole_cards, board_cards)
                
                if my_score < opp_score:  # en treys, menor puntuació = millor mà
                    wins += 1
                elif my_score == opp_score:
                    ties += 1
            
            return (wins + 0.5 * ties) / n_sim if n_sim > 0 else 0.0
        
    except Exception as e:
        print(f"⚠️ Ups! Ha passat algo amb hand_strength: {e}")
        return 0.0
    
    return 0.0


def draw_potential(hole, board):
    """Calcula aproximadament els outs que tenim → retorna un ratio entre 0 i 1. Versió simplificada."""
    # Aquesta és una heurística bastant bàsica: parelles, cartes connectades, del mateix color → més outs
    ranks = [c[0] for c in hole]
    suits = [c[1] for c in hole]
    outs = 0
    if ranks[0] == ranks[1]:
        outs += 4   # podem fer trips
    if abs("23456789TJQKA".index(ranks[0]) - "23456789TJQKA".index(ranks[1])) == 1:
        outs += 4   # escala gutshot
    if suits[0] == suits[1]:
        outs += 9   # color
    return min(1.0, outs / 20.0)
# -------------------------------------------------------------------


# ---------- constants que ens ajuden amb les recompenses -----------------
STACK_SCALE          = 1000.0

# Recompenses ben equilibrades
WIN_HAND_BONUS         = 1.0
WIN_SHOWDOWN_BONUS     = 2.0
LOSE_PENALTY           = -0.25
LOSE_EARLY_PENALTY     = -2.0
BUSTED_PENALTY         = -5.0
INITIAL_SURVIVAL_BONUS = 0.20

def calculate_survival_bonus(hand_num):
    """
    Calcula el bonus de supervivència segons el número de mans jugades.
    És positiu per menys de 20 mans, negatiu després.
    """
    if hand_num < 20:
        # Bonus positiu que va disminuint fins arribar a 0 a la mà 20
        return INITIAL_SURVIVAL_BONUS * (1 - hand_num / 20.0)
    else:
        # Bonus negatiu que va augmentant en magnitud després de 20 mans
        return max(-0.05 * (hand_num - 19), -1.0)  # Comença a -0.05 i baixa
# ---------------------------------------

class PokerTournamentEnv(AECEnv):
    """
    Entorn multiagent de Texas Hold'em amb mecàniques de torneig.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, n_agents=4):
        super().__init__()
        self.n_agents = n_agents
        self.possible_agents = [f"player_{i}" for i in range(n_agents)]
        self.agents = []  # Inicialment buit
        self._agent_selector = None
        self.agent_selection = None
        self._tournament_is_over = False
        self._first_reset = True
        self.hand_counter = 0  # Comptador de mans jugades

        # Espais - MILLORATS amb més informació rellevant
        self.action_spaces = {a: spaces.Discrete(4) for a in self.possible_agents}
        self.observation_spaces = {
            a: spaces.Dict({
                "hole_cards":      spaces.MultiDiscrete([52, 52]),      # cartes del jugador
                "community_cards": spaces.Box(-1, 51, shape=(5,),
                                            dtype=np.int64),          # cartes comunitàries
                "stack":           spaces.Box(0, 10000, shape=(1,),
                                            dtype=np.int64),          # stack del jugador
                "position":        spaces.Discrete(n_agents),           # posició a la taula
                "phase":           spaces.Discrete(5),                  # fase del joc
                # --- NOVES OBSERVACIONS CRÍTIQUES ---
                "pot_size":        spaces.Box(0, 50000, shape=(1,),
                                            dtype=np.int64),          # mida del pot
                "to_call":         spaces.Box(0, 10000, shape=(1,),
                                            dtype=np.int64),          # quantitat per fer call
                "current_bet":     spaces.Box(0, 10000, shape=(1,),
                                            dtype=np.int64),          # aposta actual
                "my_bet":          spaces.Box(0, 10000, shape=(1,),
                                            dtype=np.int64),          # la meva aposta actual
                "active_players":  spaces.Box(0, n_agents, shape=(1,),
                                            dtype=np.int64),          # jugadors actius
                "pot_odds":        spaces.Box(0, 100, shape=(1,),
                                            dtype=np.float32),        # ràtio pot odds
                "stack_ratio":     spaces.Box(0, 1, shape=(1,),
                                            dtype=np.float32),        # ràtio stack/pot
                "position_info":   spaces.Box(0, 1, shape=(3,),
                                            dtype=np.float32),        # [is_dealer, is_sb, is_bb]
                # --- features extra de força de la mà ---
                "strength":        spaces.Box(0, 1, shape=(1,),
                                            dtype=np.float32),
                "potential":       spaces.Box(0, 1, shape=(1,),
                                            dtype=np.float32),
                # --- ACTION MASKING ---
                "action_mask":     spaces.Box(0, 1, shape=(4,),
                                            dtype=np.int8),           # màscara d'accions vàlides [fold, call, raise, all-in]
            })
            for a in self.possible_agents
        }

        # Inicialitzar mètriques
        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}

    def _start_new_tournament(self):

        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}

        """Crea un torneig des de zero amb tots els jugadors i stacks inicials."""
        self.tournament = Tournament(
            self.possible_agents.copy(),        # TOTS els jugadors originals
            initial_stack=1000,
            small_blind=10,
            big_blind=20,
        )
        self._tournament_is_over = False
        # Resetejar agents eliminats per al nou torneig
        self._busted_agents = set()
        # Reiniciar comptador de mans al començar un nou torneig
        self.hand_counter = 0


    
    def _get_valid_actions(self, agent):
        """Genera la màscara d'accions vàlides per a un agent."""
        if (agent not in self.agents or 
            self.folded.get(agent, False) or 
            self.tournament.get_stack(agent) <= 0):
            # Si el agent no està actiu, només pot fer 'fold' (inactiu)
            return np.array([1, 0, 0, 0], dtype=np.int8)
        
        stack = self.tournament.get_stack(agent)
        to_call = max(0, self.current_bet - self.player_bets.get(agent, 0))
        
        # Fold sempre és vàlid
        can_fold = 1
        
        # Call és vàlid si hi ha alguna cosa que igualar i tenim fichas
        can_call = 1 if stack > 0 else 0
        
        # Raise és vàlid si tenim fichas suficient per més que call
        can_raise = 1 if stack > to_call + 10 else 0  # Raise mínim de 10
        
        # All-in sempre és vàlid si tenim fichas
        can_allin = 1 if stack > 0 else 0
        
        return np.array([can_fold, can_call, can_raise, can_allin], dtype=np.int8)

    def _empty_obs(self, agent):
        """Observació neutra – dins dels spaces – per a un agent."""
        return dict(
            hole_cards      = np.array([-1, -1], dtype=np.int64),
            community_cards = np.full((5,), -1, dtype=np.int64),
            stack           = np.array([0],       dtype=np.int64),
            position        = np.int64(0),
            phase           = np.int64(0),
                    # Noves observacions buides
        pot_size        = np.array([0], dtype=np.int64),
        to_call         = np.array([0], dtype=np.int64),
        current_bet     = np.array([0], dtype=np.int64),
        my_bet          = np.array([0], dtype=np.int64),
        active_players  = np.array([0], dtype=np.int64),
        pot_odds        = np.array([0.0], dtype=np.float32),
        stack_ratio     = np.array([0.0], dtype=np.float32),
        position_info   = np.array([0.0, 0.0, 0.0], dtype=np.float32),
        strength        = np.array([0.0], dtype=np.float32),
        potential       = np.array([0.0], dtype=np.float32),
        action_mask     = np.array([1, 0, 0, 0], dtype=np.int8),  # Només fold vàlid per defecte
        )

    def reset(self, *, seed=None, options=None):
        """RLlib crida reset() al començament de CADA episodi (mà)."""
        # — Llavor reproduïble —
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Si es el primer reset o el torneig terminó, crear nou torneig
        if self._first_reset or not hasattr(self, "tournament") or self._tournament_is_over:
            self._start_new_tournament()
            self._first_reset = False

        # Verificar que hi ha jugadors suficients per continuar
        alive = [a for a in self.possible_agents if self.tournament.get_stack(a) > 0]
        
        if len(alive) < 2:
            # Torneig terminat - crear nou torneig
            self._start_new_tournament()
            alive = [a for a in self.possible_agents if self.tournament.get_stack(a) > 0]

        #Sincronitzar agents de l'entorn amb el tournament
        self.agents = alive.copy()
        
        #Actualitzar la llista d'agents al tournament també
        if hasattr(self.tournament, 'agents'):
            # Asegurar que tournament.agents només contingui jugadors vius
            self.tournament.agents = [a for a in self.tournament.agents if a in alive]
            if not self.tournament.agents:
                self.tournament.agents = alive.copy()
        
        # Rotar posicions si no és el primer joc
        if not self._first_reset and len(alive) >= 2:
            if hasattr(self.tournament, 'rotate'):
                try:
                    self.tournament.rotate()
                except Exception as e:
                    pass

        self.rewards = {a: 0.0 for a in self.possible_agents}
        
        # Només inicialitzar _cumulative_rewards si no existeix
        if not hasattr(self, '_cumulative_rewards') or not self._cumulative_rewards:
            self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}
        
        # Mantenir un registre intern de rewards per seguiment d'aquesta mano
        self._episode_rewards = {a: 0.0 for a in self.possible_agents}
        
        # Reinicialitzar terminations i truncations per nova mà
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}

        # Marcar agents eliminats com a terminats
        for agent in self.possible_agents:
            if agent not in self.agents:
                self.terminations[agent] = True
                # Només assignar penalització per quedar eliminat UNA VEGADA per torneig
                if (agent not in getattr(self, '_busted_agents', set()) and 
                    self.tournament.get_stack(agent) == 0):
                    # NOMÉS afegir a cumulative, NO a episode_rewards en reset
                    self._cumulative_rewards[agent] += BUSTED_PENALTY
                    # I assignar la penalització per aquest step específic de RLLib
                    self.rewards[agent] = BUSTED_PENALTY
                    # Marcar com penalitzat per no repetir-ho
                    if not hasattr(self, '_busted_agents'):
                        self._busted_agents = set()
                    self._busted_agents.add(agent)


        # Inicialitzar la mano amb validació
        try:
            self.reset_hand()
        except Exception as e:
            # Fallback: reiniciar torneig des de zero
            self._start_new_tournament()
            self.agents = [a for a in self.possible_agents if self.tournament.get_stack(a) > 0]
            self.reset_hand()

        # Retornar observacions inicials
        observations = {}
        for agent in self.possible_agents:
            if agent in self.agents:
                observations[agent] = self.observe(agent)
            else:
                observations[agent] = self._empty_obs(agent)

        return observations

    def reset_hand(self):
        """Reparte cartes, assigna blinds i ordena el turno — NO recrea Tournament."""
        if len(self.agents) < 2:
            return
            
        # Incrementar el comptador de mans
        self.hand_counter += 1
            
        # 1) barrejar i repartir cartes hole i community
        self.deck = Deck()
        self.deck.shuffle()
        self.hole_cards = {a: self.deck.draw(2) for a in self.agents}
        self.community_cards = []
        
        # 2) resetejar aposts
        self.phase = 0
        self.pot = 0
        self.current_bet = 0
        self.folded = {a: False for a in self.possible_agents}
        self.player_bets = {a: 0 for a in self.possible_agents}
        self.raises_this_round = 0
        self.last_actions = {a: "Esperant" for a in self.possible_agents}
        self.has_acted = {a: False for a in self.possible_agents}

        # 3) assignar blinds - AMB VALIDACIÓ
        try:
            self.dealer, self.small_blind, self.big_blind, self.blinds = \
                self.tournament.assign_blinds()
        except Exception as e:
            # Pla de contingència: usar els primers dos agents disponibles
            if len(self.agents) >= 2:
                self.dealer = self.agents[0]
                self.small_blind = self.agents[0]
                self.big_blind = self.agents[1]
                self.blinds = {
                    self.small_blind: self.tournament.small_blind,
                    self.big_blind: self.tournament.big_blind
                }
            else:
                return
        
        for a, b in self.blinds.items():
            self.pot += b
            self.player_bets[a] = b
            self.current_bet = max(self.current_bet, b)
            
        # 4) construir nou ordre de torn després del BB - AMB VALIDACIÓ
        seats = list(self.agents)
        
        #Verificar que big_blind està en seats
        if self.big_blind not in seats:
            # Pla de contingència: començar des del primer jugador
            self.agents = [a for a in seats if not self.folded[a]]
            self.hand_order = list(self.agents)
        else:
            try:
                bb_idx = seats.index(self.big_blind)
                after_bb = seats[bb_idx+1:] + seats[:bb_idx+1]
                self.agents = [a for a in after_bb if not self.folded[a]]
                self.hand_order = list(self.agents)
            except ValueError as e:
                # Pla de contingència: usar l'ordre actual
                self.agents = [a for a in seats if not self.folded[a]]
                self.hand_order = list(self.agents)
        
        if self.agents:
            self._agent_selector = agent_selector(self.agents)
            self.agent_selection = self._agent_selector.reset()

    def observe(self, agent):
        def c2i(card):  # 0-51
            ranks, suits = "23456789TJQKA", "CDHS"
            return ranks.index(card[0]) * 4 + suits.index(card[1])

        # Si l'agent no té cartes (eliminat), retornar observació buida
        if agent not in self.hole_cards:
            return self._empty_obs(agent)

        # Informació bàsica
        hole = np.array([c2i(c) for c in self.hole_cards[agent]], dtype=np.int64)
        comm = np.full((5,), -1, dtype=np.int64)
        for i, c in enumerate(self.community_cards):
            comm[i] = c2i(c)

        stack = np.array([self.tournament.get_stack(agent)], dtype=np.int64)
        pos = np.int64(self.possible_agents.index(agent))
        phase = np.int64(self.phase)

        # --- NOVA INFORMACIÓ CRUCIAL ---
        pot_size = np.array([self.pot], dtype=np.int64)
        current_bet = np.array([self.current_bet], dtype=np.int64)
        my_bet = np.array([self.player_bets.get(agent, 0)], dtype=np.int64)
        to_call = np.array([max(0, self.current_bet - self.player_bets.get(agent, 0))], dtype=np.int64)
        
        # Jugadors actius (no s'han retirat ni eliminat)
        active_players = len([a for a in self.agents 
                             if not self.folded.get(a, False) and 
                             self.tournament.get_stack(a) > 0])
        active_players = np.array([active_players], dtype=np.int64)
        
        # Pot odds (ràtio d'aposta necessària vs mida del pot)
        if to_call[0] > 0 and self.pot > 0:
            pot_odds_val = min(100.0, (to_call[0] / self.pot) * 100)
        else:
            pot_odds_val = 0.0
        pot_odds = np.array([pot_odds_val], dtype=np.float32)
        
        # Ràtio stack vs pot (per decisions de risc)
        if self.pot > 0:
            stack_ratio_val = min(1.0, stack[0] / self.pot)
        else:
            stack_ratio_val = 1.0 if stack[0] > 0 else 0.0
        stack_ratio = np.array([stack_ratio_val], dtype=np.float32)
        
        # Informació de posició (dealer, small blind, big blind)
        position_info = np.array([
            1.0 if agent == getattr(self, 'dealer', None) else 0.0,
            1.0 if agent == getattr(self, 'small_blind', None) else 0.0,
            1.0 if agent == getattr(self, 'big_blind', None) else 0.0
        ], dtype=np.float32)

        # --- calcular força i potencial ---
        strength  = np.array([hand_strength(self.hole_cards[agent],
                                            self.community_cards)], dtype=np.float32)
        potential = np.array([draw_potential(self.hole_cards[agent],
                                             self.community_cards)], dtype=np.float32)

        # Action mask per accions vàlides
        action_mask = self._get_valid_actions(agent)

        return {
            "hole_cards": hole,
            "community_cards": comm,
            "stack": stack,
            "position": pos,
            "phase": phase,
            "pot_size": pot_size,
            "to_call": to_call,
            "current_bet": current_bet,
            "my_bet": my_bet,
            "active_players": active_players,
            "pot_odds": pot_odds,
            "stack_ratio": stack_ratio,
            "position_info": position_info,
            "strength": strength,
            "potential": potential,
            "action_mask": action_mask,
        }
    
    def _end_hand(self, truncated=False):
        """Acaba la mà i prepara per la següent."""
        
        # bonus per seguir viu
        for a in self.possible_agents:
            if self.tournament.get_stack(a) > 0:
                self._episode_rewards[a] += calculate_survival_bonus(self.hand_counter)
                self._cumulative_rewards[a] += calculate_survival_bonus(self.hand_counter)
                self.rewards[a] = self.rewards.get(a, 0.0) + calculate_survival_bonus(self.hand_counter)


        # Marcar com acabat/truncat només per agents actius d'aquesta mà
        for agent in self.agents:
            if truncated:
                self.truncations[agent] = True
            else:
                self.terminations[agent] = True

        # Netejar el torneig de jugadors sense fitxes
        self.tournament.drop_busted()
        
        # Verificar si el torneig realment ha d'acabar
        alive_players = [a for a in self.possible_agents if self.tournament.get_stack(a) > 0]
        
        if len(alive_players) <= 1:
            # Torneig realment acabat - només queda un jugador o cap
            self._tournament_is_over = True
            # Marcar TOTS els agents com acabats
            for agent in self.possible_agents:
                self.terminations[agent] = True
            # Netejar llista d'agents actius
            self.agents = []
            pass
        else:
            # El torneig continua - només marcar agents d'aquesta mà com acabats
            # però NO marcar _tournament_is_over = True
            # Els agents vius continuaran en la següent mà
            
            # Només netejar la llista d'agents actius d'aquesta mà
            self.agents = []

    def step(self, action):
        if not self.agents:
            return
            
        current = self.agent_selection

        # Saltar si no hi ha acció o ja s'ha retirat/acabat
        if (action is None or 
            self.folded.get(current, False) or 
            self.terminations.get(current, False) or
            current not in self.agents):
            
            if current in self.agents:
                self.last_actions[current] = "Salta" if action is not None else "Cap"
                self.agent_selection = self._agent_selector.next()
            return 
        
        #Validar action masking - convertir accions invàlides en fold
        valid_actions = self._get_valid_actions(current)
        if valid_actions[action] == 0:  # Acció invàlida
            action = 0  # Convertir a fold
            print(f"⚠️ Acció invàlida convertida a fold per {current}") 
        
        # Processar acció
        act_str = {0: "Fold", 1: "Call", 2: "Puja mínima", 3: "All-in"}

        # Resetejar rewards de step abans de processar acció
        self.rewards = {a: 0.0 for a in self.possible_agents}
        
        # Actualitzar infos amb informació de la mà i survival bonus
        for a in self.possible_agents:
            self.infos[a] = {
                "hand_counter": self.hand_counter,
                "survival_bonus": calculate_survival_bonus(self.hand_counter)
            }

        stack = self.tournament.get_stack(current)
        if stack > 0:
            if action == 0:  # Fold
                self.folded[current] = True

            elif action == 1:  # Call
                to_call = self.current_bet - self.player_bets[current]
                pay = min(to_call, stack)
                self.tournament.update_stack(current, -pay)
                self.pot += pay
                self.player_bets[current] += pay

            elif action in (2, 3):  # Raise or All-in
                to_call = self.current_bet - self.player_bets[current]
                base = 10 if action == 2 else stack
                desired = to_call + base
                pay = min(desired, stack)
                self.tournament.update_stack(current, -pay)
                self.pot += pay
                self.player_bets[current] += pay

                # càstig per all-in prematur (independentment del resultat)
                if action == 3 and self.phase < 2:   # Pre-flop o Flop
                    self._episode_rewards[current] += LOSE_EARLY_PENALTY
                    self._cumulative_rewards[current] += LOSE_EARLY_PENALTY
                    self.rewards[current] += LOSE_EARLY_PENALTY

                old = self.current_bet
                self.current_bet = max(self.current_bet, self.player_bets[current])
                if self.current_bet > old:
                    for p in self.agents:
                        if (p != current and 
                            not self.folded[p] and 
                            self.tournament.get_stack(p) > 0):
                            self.has_acted[p] = False
            
        self.has_acted[current] = True
        self.last_actions[current] = act_str.get(action, "?")

    
        # Avançar al següent agent
        if self.agents:
            self.agent_selection = self._agent_selector.next()

        # Verificar fi de ronda d'apostes
        active = [a for a in self.agents 
                 if not self.folded[a] and self.tournament.get_stack(a) >= 0]

        # Marcar com actuats els jugadors all-in
        for p in active:
            if self.tournament.get_stack(p) == 0:
                self.has_acted[p] = True
                self.last_actions[p] = "All-in"  # Actualitzar l'última acció a All-in
        
        # Verificar si la ronda d'apostes ha acabat
        betting_complete = (
            len(active) <= 1 or 
            (all(self.has_acted[a] for a in active) and 
             all(self.player_bets[a] >= self.current_bet or 
                 self.tournament.get_stack(a) == 0 for a in active))
        )

        if betting_complete:
            # Reset d'apostes per la següent ronda
            for a in active:
                self.has_acted[a] = False
                self.player_bets[a] = 0
            self.current_bet = 0

            # Si queda un sol jugador actiu: guanya
            if len(active) <= 1:
                if active:
                    winner = active[0]
                    pot_reward = self.pot / STACK_SCALE
                    self.tournament.update_stack(winner, self.pot)
                    
                    
                    self.pot = 0
                    
                    for a in self.agents:
                        if a == winner:
                            base_reward = pot_reward + WIN_HAND_BONUS
                            
                            self._episode_rewards[a] += base_reward
                            self._cumulative_rewards[a] += base_reward
                            self.rewards[a] = base_reward

                        elif not self.folded[a]:
                            self._episode_rewards[a] += LOSE_PENALTY
                            self._cumulative_rewards[a] += LOSE_PENALTY
                            self.rewards[a] = LOSE_PENALTY

                
                self._end_hand()
                return

            # Avanç de fase o showdown
            if self.phase == 3:  # River completat, anar a showdown
                self._handle_showdown(active)
                return
            else:
                # Avançar a la següent fase
                if self.phase == 0:  # Pre-flop -> Flop
                    self.phase = 1
                    self.community_cards.extend(self.deck.draw(3))
                elif self.phase == 1:  # Flop -> Turn
                    self.phase = 2
                    self.community_cards.extend(self.deck.draw(1))
                elif self.phase == 2:  # Turn -> River
                    self.phase = 3
                    self.community_cards.extend(self.deck.draw(1))

                # Reordenar agents per la nova ronda
                seats = list(self.hand_order)
                try:
                    #Verificar que big_blind existeix en seats
                    if self.big_blind not in seats:
                        # Pla de contingència: usar l'ordre actual d'active
                        self.agents = active.copy()
                    else:
                        bb_idx = seats.index(self.big_blind)
                        ordered = seats[bb_idx+1:] + seats[:bb_idx+1]
                        self.agents = [a for a in ordered 
                                     if not self.folded[a] and a in active]
                    
                    if self.agents:
                        self._agent_selector = agent_selector(self.agents)
                        self.agent_selection = self._agent_selector.reset()
                    else:
                        self._end_hand()
                        
                except (ValueError, IndexError) as e:
                    self._end_hand()

    def _handle_showdown(self, active):
        """Gestiona el showdown al final de la mà."""
        candidates = [p for p in active if not self.folded[p]]
        if not candidates:
            self._end_hand()
            return

        if len(candidates) == 1:
            winner = candidates[0]
        else:
            # Avaluar mans
            evaluator = Evaluator()
            scores = {}
            for a in candidates:
                try:
                    hole = [Card.new(c[0] + c[1].lower()) for c in self.hole_cards[a]]
                    board = [Card.new(c[0] + c[1].lower()) for c in self.community_cards]
                    scores[a] = evaluator.evaluate(hole, board)
                except:
                    scores[a] = 7463  # Pitjor mà possible en cas d'error

            winner = min(scores, key=scores.get)

        # Repartir el pot
        pot_reward = self.pot / STACK_SCALE
        self.tournament.update_stack(winner, self.pot)
        self.pot = 0
        
        # Assignar rewards finals
        for a in self.agents:
            if a == winner:
                reward = pot_reward + WIN_SHOWDOWN_BONUS
                self._episode_rewards[a] += reward
                self._cumulative_rewards[a] += reward
                self.rewards[a] = reward  # Només per aquest step

            elif not self.folded[a]:
                self._episode_rewards[a] += LOSE_PENALTY
                self._cumulative_rewards[a] += LOSE_PENALTY
                self.rewards[a] = LOSE_PENALTY  # Només per aquest step

        self._end_hand()


    def render(self, mode="human"):
        """Imprimeix l'estat actual per consola."""
        if not hasattr(self, 'phase'):
            return
            
        phases = ["Pre-Flop", "Flop", "Turn", "River", "Showdown"]
        current_survival_bonus = calculate_survival_bonus(self.hand_counter)
        print(f"=== Mà #{self.hand_counter} | {phases[self.phase]} | Pot: {self.pot} | Aposta actual: {self.current_bet} ===")
        print(f"=== Bonus supervivència: {current_survival_bonus:.2f} {'(positiu)' if current_survival_bonus > 0 else '(negatiu)'} ===")
        if self.community_cards:
            print("Comunitàries:", self.community_cards)
        for a in self.agents:
            if a in self.hole_cards:
                print(
                    f"{a:8s} | Cartes: {self.hole_cards[a]} | "
                    f"Stack: {self.tournament.get_stack(a):7.0f} | "
                    f"Apostat: {self.player_bets[a]:4.0f} | "
                    f"Última: {self.last_actions[a]:12s} | "
                    f"Recompensa: {self.rewards.get(a,0):5.2f} | "
                    f"Retirat: {self.folded[a]}"
                )
        print("=" * 70)

    def seed(self, seed=None):
        """Perquè RLlib env‐checker pugui cridar env.seed(x)."""
        random.seed(seed)
        np.random.seed(seed)

    def close(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
