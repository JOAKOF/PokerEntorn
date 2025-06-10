from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.appo import APPOConfig
import numpy as np
import random
import argparse
import sys
from ray import tune
from poker_env import PokerTournamentEnv
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

# registre de l'entorn
tune.register_env("poker_tournament", lambda config: PettingZooEnv(PokerTournamentEnv(
    n_agents=config.get("n_players", 4)
    )))





# Configuració DQN
dqn_cfg = (DQNConfig()
             .environment("poker_tournament")
             .rollouts(num_rollout_workers=0)
             .resources(num_gpus=0))
dqn = dqn_cfg.build()
dqn.restore(r"C:\Users\HP\ray_results\DQN_2025-06-09_09-58-36\DQN_poker_tournament_8beeb_00000_0_2025-06-09_09-58-36\checkpoint_000008")



# Configuració APPO
appo_cfg = (APPOConfig()
             .environment("poker_tournament")
             .rollouts(num_rollout_workers=0)
             .resources(num_gpus=0))
appo = appo_cfg.build()
# TODO: actualitzar amb el path correcte del checkpoint d'APPO
appo.restore(r"C:\Users\HP\ray_results\APPO_2025-06-09_14-33-16\APPO_poker_tournament_eafcf_00000_0_2025-06-09_14-33-16\checkpoint_000002")



# Crear entorn per comparar algoritmes
def create_test_env(n_agents=4):
    """Crea un entorn de prova per comparar algoritmes"""
    env = PokerTournamentEnv(n_agents=n_agents)
    return env

def play_algorithms(dqn_agent, appo_agent, num_games=5, player_config=None):
    """Fa que DQN i APPO juguin entre ells en torneigs complets"""
    # Configuració per defecte si no s'especifica
    if player_config is None:
        player_config = {
            "player_0": ("DQN", dqn_agent),
            "player_1": ("APPO", appo_agent),
            "player_2": ("Random", None),
            "player_3": ("Random", None)
        }
    
    n_players = len(player_config)
    
    print("=== INICIANT COMPARACIÓ D'ALGORITMES ===")
    print(f"Torneig amb {n_players} jugadors - {num_games} partides")
    print("-" * 50)
    
    # Estadístiques - cada jugador per separat
    wins = {}
    total_rewards = {}
    action_stats = {}
    hand_stats = {}
    position_stats = {}
    
    for player, (algo, _) in player_config.items():
        player_key = f"{player} ({algo})"
        wins[player_key] = 0
        total_rewards[player_key] = []
        
        # Estadístiques d'accions
        action_stats[player_key] = {
            'fold': 0, 'call': 0, 'raise': 0, 'allin': 0, 'total_actions': 0
        }
        
        # Estadístiques de mans i comportament
        hand_stats[player_key] = {
            'hands_played': 0, 'tournaments_participated': 0,
            'eliminations': 0, 'final_positions': [],
            'chips_won': [], 'chips_lost': [],
            'showdowns_reached': 0, 'showdowns_won': 0
        }
        
        # Estadístiques de posició (early, middle, late)
        position_stats[player_key] = {
            'early_position_actions': 0, 'middle_position_actions': 0, 
            'late_position_actions': 0, 'dealer_actions': 0
        }
    
    for game in range(num_games):
        print(f"\n🏆 TORNEIG {game + 1}/{num_games}")
        print("=" * 40)
        
        # Crear entorn per aquest torneig
        env = create_test_env(n_players)
        env.reset()
        
        # Usar la configuració de jugadors passada
        agent_assignments = player_config
        
        print("🎯 Participants del torneig:")
        for player, (algo, _) in agent_assignments.items():
            stack = env.tournament.get_stack(player)
            print(f"  {player}: {algo} (Stack inicial: {stack})")
            # Registrar participació en torneig
            player_key = f"{player} ({algo})"
            hand_stats[player_key]['tournaments_participated'] += 1
        
        hand_number = 1
        eliminated_players = set()
        
        # Bucle principal del torneig (basat en main.py)
        while True:
            print(f"\n=== ✋ Mà #{hand_number} ===")
            
            # Jugar la mà actual fins que acabi
            step_in_hand = 0
            while not all(env.terminations.values()) and step_in_hand < 200:
                current_agent = env.agent_selection
                 
                if current_agent and not env.terminations.get(current_agent, False):
                    obs = env.observe(current_agent)
                    algo_name, algo_agent = agent_assignments[current_agent]
                     
                    # Mostrar estat del joc abans de l'acció
                    print(f"\n  --- Torn de {current_agent} ({algo_name}) ---")
                     
                    # Informació de l'estat actual
                    phase_names = ["Pre-flop", "Flop", "Turn", "River", "Showdown"]
                    current_phase = phase_names[min(env.phase, 4)] if hasattr(env, 'phase') else "Unknown"
                    pot = getattr(env, 'pot', 0)
                    current_bet = getattr(env, 'current_bet', 0)
                     
                    print(f"    🎯 Fase: {current_phase} | 💰 Pot: {pot} | 🎲 Aposta actual: {current_bet}")
                     
                    # Cartes comunitàries (si n'hi ha)
                    if hasattr(env, 'community_cards') and env.community_cards:
                        print(f"    🃏 Taula: {env.community_cards}")
                     
                    # Informació del jugador
                    stack = env.tournament.get_stack(current_agent)
                    player_bet = getattr(env, 'player_bets', {}).get(current_agent, 0)
                     
                    # Mostrar cartes hole només per DQN i APPO (no per Random)
                    if algo_name in ["DQN", "APPO"] and hasattr(env, 'hole_cards') and current_agent in env.hole_cards:
                        hole_cards = env.hole_cards[current_agent]
                        print(f"    🂠 {current_agent}: {hole_cards} | Stack: {stack} | Apostat: {player_bet}")
                    else:
                        print(f"    👤 {current_agent}: Stack: {stack} | Apostat: {player_bet}")
                    
                    # Mostrar estat d'altres jugadors
                    other_players = []
                    for p, (a, _) in agent_assignments.items():
                        if p != current_agent:
                            p_stack = env.tournament.get_stack(p)
                            p_bet = getattr(env, 'player_bets', {}).get(p, 0)
                            folded = getattr(env, 'folded', {}).get(p, False)
                            status = "💀 Fold" if folded else f"💵 {p_bet}"
                            other_players.append(f"{p}({a}): {p_stack} ({status})")
                    
                    if other_players:
                        print(f"    🎭 Altres: {' | '.join(other_players)}")
                    
                    try:
                        if algo_name == "DQN":
                            # Usar DQN per decidir
                            action_result = algo_agent.compute_single_action(obs, explore=False)
                            action = action_result[0] if isinstance(action_result, (list, tuple)) else action_result
                            
                        elif algo_name == "APPO":
                            # Usar APPO per decidir
                            action_result = algo_agent.compute_single_action(obs, explore=False)
                            action = action_result[0] if isinstance(action_result, (list, tuple)) else action_result
                            
                        else:  # Random
                            # Acció amb pesos (com en main.py)
                            action = random.randint(0, 3)
                            
                    except Exception as e:
                        print(f"    ❌ ERROR per {current_agent} ({algo_name}): {e}")
                        action = env.action_space(current_agent).sample()
                    
                    # Mostrar l'acció decidida
                    action_names = ["Fold", "Call", "MinRaise", "All-in"]
                    action_emojis = ["😔", "✋", "⬆️", "🚀"]
                    
                    # Calcular quantitat a apostar per mostrar
                    to_call = current_bet - player_bet
                    if action == 1:  # Call
                        amount = min(to_call, stack)
                        action_detail = f"Call {amount}"
                    elif action == 2:  # MinRaise
                        amount = min(to_call + 10, stack)
                        action_detail = f"Raise a {player_bet + amount}"
                    elif action == 3:  # All-in
                        action_detail = f"All-in {stack}"
                    else:  # Fold
                        action_detail = "Fold"
                    
                    print(f"    🎬 ACCIÓ: {action_emojis[action]} {action_names[action]} ({action_detail})")
                    
                    # Verificar quina acció realment s'executa
                    stack_before = stack
                    bet_before = player_bet
                    folded_before = getattr(env, 'folded', {}).get(current_agent, False)
                    
                    # Registrar estadístiques de posició
                    n_players_in_game = len([p for p in agent_assignments.keys() 
                                           if env.tournament.get_stack(p) > 0])
                    agent_index = list(agent_assignments.keys()).index(current_agent)
                    
                    if current_agent == getattr(env, 'dealer', None):
                        position_stats[player_key]['dealer_actions'] += 1
                    elif agent_index < n_players_in_game // 3:
                        position_stats[player_key]['early_position_actions'] += 1
                    elif agent_index < 2 * n_players_in_game // 3:
                        position_stats[player_key]['middle_position_actions'] += 1
                    else:
                        position_stats[player_key]['late_position_actions'] += 1
                    
                    env.step(action)
                    
                    # Verificar quina acció realment es va executar
                    stack_after = env.tournament.get_stack(current_agent)
                    bet_after = getattr(env, 'player_bets', {}).get(current_agent, 0)
                    folded_after = getattr(env, 'folded', {}).get(current_agent, False)
                    new_pot = getattr(env, 'pot', 0)
                    
                    # Registrar estadístiques d'acció basat en el que realment va passar
                    player_key = f"{current_agent} ({algo_name})"
                    action_stats[player_key]['total_actions'] += 1
                    hand_stats[player_key]['hands_played'] += 1
                    
                    # Determinar l'acció real executada
                    if folded_after and not folded_before:
                        # Es va fer fold (real o per acció invàlida)
                        action_stats[player_key]['fold'] += 1
                        if action != 0:
                            print(f"    ⚠️ Acció {action_names[action]} es va convertir en Fold (invàlida)")
                    elif stack_after == 0 and stack_before > 0:
                        # All-in (va gastar tot l'stack)
                        action_stats[player_key]['allin'] += 1
                    elif bet_after > bet_before:
                        # Raise o call (va apostar algo)
                        bet_increase = bet_after - bet_before
                        if bet_increase == (current_bet - bet_before):
                            # Només va igualar l'aposta = call
                            action_stats[player_key]['call'] += 1
                        else:
                            # Va apostar més del necessari = raise
                            action_stats[player_key]['raise'] += 1
                    else:
                        # No va apostar res addicional = check/call de 0
                        action_stats[player_key]['call'] += 1
                    
                    # Mostrar resultat de l'acció
                    stack_change = stack_after - stack_before
                    pot_change = new_pot - pot
                    
                    if stack_change != 0:
                        print(f"    📊 Resultat: Stack {stack_before} → {stack_after} ({stack_change:+}), Pot {pot} → {new_pot} ({pot_change:+})")
                    
                else:
                    env.step(None)  # Skip agent terminat
                
                step_in_hand += 1

            
            # Mostrar stacks al final de la mà
            print("\n📊 Stacks després de la mà:")
            survivors = []
            eliminated_this_hand = []
            
            # Primer, identificar supervivents i eliminats
            for player, (algo, _) in agent_assignments.items():
                stack = env.tournament.get_stack(player)
                
                if stack > 0:
                    survivors.append((player, algo, stack))
                elif player not in eliminated_players:
                    eliminated_this_hand.append((player, algo))
                    eliminated_players.add(player)
            
            # Assignar posicions finals als eliminats (de millor a pitjor per stack restant)
            if eliminated_this_hand:
                # Els eliminats en aquesta mà tenen posició = número de supervivents + posició relativa
                base_position = len(survivors) + 1
                
                # Si hi ha varis eliminats simultàniament, assignar la mateixa posició
                for i, (player, algo) in enumerate(eliminated_this_hand):
                    player_key = f"{player} ({algo})"
                    final_position = base_position + i
                    hand_stats[player_key]['eliminations'] += 1
                    hand_stats[player_key]['final_positions'].append(final_position)
            
            # Mostrar estats
            for player, (algo, _) in agent_assignments.items():
                stack = env.tournament.get_stack(player)
                status = "💀 ELIMINAT" if stack <= 0 else ""
                reward = env._cumulative_rewards.get(player, 0) if hasattr(env, '_cumulative_rewards') else 0
                print(f"  {player} ({algo}): {stack} fitxes, Reward: {reward:.2f} {status}")
            
            # Verificar si el torneig ha de continuar
            if len(survivors) < 2:
                if len(survivors) == 1:
                    winner_player, winner_algo, winner_stack = survivors[0]
                    print(f"\n🏆 ¡TORNEIG TERMINAT! Guanyador: {winner_player} ({winner_algo}) amb {winner_stack} fitxes")
                    
                    # Registrar victòria i recompenses per TOTS els jugadors
                    for player, (algo, _) in agent_assignments.items():
                        player_key = f"{player} ({algo})"
                        reward = env._cumulative_rewards.get(player, 0) if hasattr(env, '_cumulative_rewards') else 0
                        total_rewards[player_key].append(reward)
                    
                    # El guanyador obté la victòria i posició 1
                    winner_key = f"{winner_player} ({winner_algo})"
                    wins[winner_key] += 1
                    hand_stats[winner_key]['final_positions'].append(1)
                    
                elif len(survivors) == 0:
                    print("\n⚠️ No queden jugadors amb fitxes. Error en el torneig!")
                    # Registrar recompenses fins i tot en cas d'error
                    for player, (algo, _) in agent_assignments.items():
                        player_key = f"{player} ({algo})"
                        reward = env._cumulative_rewards.get(player, 0) if hasattr(env, '_cumulative_rewards') else 0
                        total_rewards[player_key].append(reward)
                break
            
            # Continuar amb nova mà
            print(f"\n🔄 Continua el torneig amb {len(survivors)} jugadors...")
                        
            env.reset()            
            hand_number += 1
        
        print(f"\n📈 Torneig {game + 1} completat amb {hand_number} mans jugades")
    
    # Estadístiques finals de tots els torneigs
    print("\n" + "=" * 60)
    print("🏆 ESTADÍSTIQUES FINALS DE TORNEIGS")
    print("=" * 60)
    
    print(f"🥇 Victòries de torneigs:")
    # Ordenar per número de victòries (descendent)
    sorted_wins = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    for player, count in sorted_wins:
        percentage = (count / num_games) * 100
        print(f"  {player}: {count}/{num_games} torneigs guanyats ({percentage:.1f}%)")
    
    print(f"\n💰 Recompenses promig per torneig:")
    # Calcular estadístiques de recompenses usant cumulative_rewards de cada torneig
    for player_key, rewards in total_rewards.items():
        if not rewards:
            continue
        avg_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)
        print(f"  {player_key}: Promig {avg_reward:.2f} (Max: {max_reward:.2f}, Min: {min_reward:.2f})")

# Configuracions predefinides de torneigs
def config_dqn_vs_appo_vs_2_random():
    """DQN vs APPO vs 2 jugadors random (4 jugadors total)"""
    return {
        "player_0": ("DQN", dqn),
        "player_1": ("APPO", appo), 
        "player_2": ("Random", None),
        "player_3": ("Random", None)
    }

def config_dqn_vs_appo_only():
    """Només DQN vs APPO (heads-up)"""
    return {
        "player_0": ("DQN", dqn),
        "player_1": ("APPO", appo)
    }

def config_multiple_dqn_vs_multiple_appo():
    """2 DQN vs 2 APPO (4 jugadors total)"""
    return {
        "player_0": ("DQN_1", dqn),
        "player_1": ("DQN_2", dqn),
        "player_2": ("APPO_1", appo),
        "player_3": ("APPO_2", appo)
    }

def config_ai_vs_random_battle():
    """2 IA vs 2 Random (4 jugadors total)"""
    return {
        "player_0": ("DQN", dqn),
        "player_1": ("APPO", appo),
        "player_2": ("Random_1", None),
        "player_3": ("Random_2", None)
    }

if __name__ == "__main__":
    # Configuració per defecte: DQN vs APPO vs 2 Random
    print("🎮 Executant comparació DQN vs APPO...")
    
    config = config_dqn_vs_appo_vs_2_random()
    
    try:
        play_algorithms(dqn, appo, num_games=10, player_config=config)
    except Exception as e:
        print(f"❌ Error durant l'execució: {e}")
        import traceback
        traceback.print_exc()

