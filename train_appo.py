import ray, wandb
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune import Tuner
from ray import air
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from poker_env import PokerTournamentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np
import torch

def unwrap_until_attr(env, attr_name):
    """
    Retorna (objecte_amb_attr, valor_del_attr)
    o (None, None) si no el troba.
    """
    visited = set()
    while env is not None and id(env) not in visited:
        visited.add(id(env))
        if hasattr(env, attr_name):
            return env, getattr(env, attr_name)
        # Els wrappers de PettingZoo / Gym acostumen a dir-se .env o .aec_env
        env = getattr(env, "env", None) or getattr(env, "aec_env", None)
    return None, None

# Callback personalitzat per mètriques
class PokerCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # Comptadors per winrate de torneigs
        self.tournament_wins = {}
        self.tournament_count = 0
        self.current_tournament_players = set()
        # Seguiment de finestres lliscants
        self.recent_winners = []
        self.window_size = 100
        # Comptadors d'accions per jugador
        self.action_counts = {}
        self.total_actions = 0
        # Comptador d'accions per torneig actual
        self.current_tournament_actions = {}

    
    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        """Registra mètriques en cada pas de l'episodi"""
        sub_env = base_env.get_sub_environments()[0]
        raw_env, _ = unwrap_until_attr(sub_env, "tournament")
        
        if raw_env and hasattr(raw_env, 'agent_selection'):
            agent = raw_env.agent_selection
            
            if agent and agent in raw_env.agents:
                if agent not in self.action_counts:
                    self.action_counts[agent] = 0
                self.action_counts[agent] += 1
                self.total_actions += 1
                
                if agent not in self.current_tournament_actions:
                    self.current_tournament_actions[agent] = 0
                self.current_tournament_actions[agent] += 1
                
                for a, count in self.current_tournament_actions.items():
                    total = sum(self.current_tournament_actions.values())
                    percentage = (count / total) * 100 if total > 0 else 0
                    episode.custom_metrics[f"{a}_action_percentage"] = float(percentage)
                
                episode.custom_metrics["total_actions_count"] = int(sum(self.current_tournament_actions.values()))
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """Procesa resultados del entrenamiento para extraer métricas adicionales"""
        # Log GPU usage si está disponible
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated(0) / 1024 / 1024
            result["gpu_mem_usage_mb"] = gpu_usage

        
        # ✅ NUEVA MÉTRICA: Tournament Winrate
        if self.tournament_count > 0:
            # Winrate total (desde el inicio)
            for agent in self.tournament_wins.keys():
                wins = self.tournament_wins[agent]
                winrate = (wins / self.tournament_count) * 100
                result[f"{agent}_tournament_winrate_total"] = float(winrate)
                result[f"{agent}_tournament_wins"] = int(wins)
            
            # Winrate reciente (últimos 100 torneos)
            if len(self.recent_winners) >= 10:  # Solo si tenemos suficientes datos
                recent_counts = {}
                for winner in self.recent_winners:
                    recent_counts[winner] = recent_counts.get(winner, 0) + 1
                
                recent_total = len(self.recent_winners)
                for agent in self.tournament_wins.keys():
                    recent_wins = recent_counts.get(agent, 0)
                    recent_winrate = (recent_wins / recent_total) * 100
                    result[f"{agent}_tournament_winrate_recent"] = float(recent_winrate)
                
                result["recent_tournaments_window"] = int(recent_total)
            
            result["total_tournaments_played"] = int(self.tournament_count)
            # Calcular winrate promedio correctamente
            if self.tournament_wins:
                total_wins = sum(self.tournament_wins.values())
                avg_winrate = (total_wins / len(self.tournament_wins)) / self.tournament_count * 100
                result["average_tournament_winrate"] = float(avg_winrate)
            else:
                result["average_tournament_winrate"] = 0.0
            
        
        
        # Métricas de entrenamiento por política
        for policy_id in result["info"]["learner"]:
            metrics = result["info"]["learner"][policy_id]
            if "learner_stats" in metrics:
                stats = metrics["learner_stats"]
                for key in ["vf_explained_variance", "policy_loss", "vf_loss", "total_loss"]:
                    if key in stats:
                        result[f"{policy_id}/{key}"] = stats[key]

    def on_episode_end(self, *, worker, base_env, episode, **kwargs):
        """Loggear métricas al final de cada episodio"""
        # ---- 1) Localizar el AEC original usando unwrap_until_attr ---
        sub_env = base_env.get_sub_environments()[0]

        raw_env, episode_rewards = unwrap_until_attr(sub_env, "_episode_rewards")
        raw_env, cumulative     = unwrap_until_attr(sub_env, "_cumulative_rewards")

        if episode_rewards is None:
            return

        # ---- 2) Reward de la mano (episodio RLlib) ----
        for agent, r in episode_rewards.items():
            episode.custom_metrics[f"{agent}_hand_reward"] = r

        # ---- 3) Reward acumulado a nivel de torneo ----
        if cumulative:
            for agent, r in cumulative.items():
                episode.custom_metrics[f"{agent}_cumulative_reward"] = r
            
            total_rewards = sum(cumulative.values())
            max_reward = max(cumulative.values()) if cumulative else 0
            min_reward = min(cumulative.values()) if cumulative else 0
            reward_std = np.std(list(cumulative.values())) if cumulative else 0
            
            episode.custom_metrics.update({
                "total_tournament_reward": total_rewards,
                "max_agent_reward": max_reward,
                "min_agent_reward": min_reward,
                "reward_std": reward_std,
            })

        # ---- 4) Métricas del torneo ----
        if hasattr(raw_env, 'tournament'):
            tournament = raw_env.tournament
            agents = getattr(tournament, 'agents', [])
            if agents:
                stacks = [tournament.get_stack(a) for a in agents]
                alive_players = len([a for a in agents if tournament.get_stack(a) > 0])
                tournament_finished = getattr(raw_env, "_tournament_is_over", False)
                
                episode.custom_metrics.update({
                    "active_players": alive_players,
                    "max_stack": max(stacks) if stacks else 0,
                    "min_stack": min(stacks) if stacks else 0,
                    "mean_stack": np.mean(stacks) if stacks else 0,
                    "stack_std": np.std(stacks) if stacks else 0,
                    "tournament_finished": 1 if tournament_finished else 0,
                    "elimination_count": len(agents) - alive_players,
                })
                
                # ✅ NUEVA LÓGICA: Detectar ganadores de torneo
                if tournament_finished and alive_players <= 1:
                    # Inicializar contadores si es necesario
                    for agent in agents:
                        if agent not in self.tournament_wins:
                            self.tournament_wins[agent] = 0
                    
                    # Encontrar el ganador (el que tiene fichas > 0)
                    winner = None
                    for agent in agents:
                        if tournament.get_stack(agent) > 0:
                            winner = agent
                            break
                    
                    # Actualizar estadísticas
                    if winner:
                        self.tournament_wins[winner] += 1
                        
                        # Agregar a ventana deslizante
                        self.recent_winners.append(winner)
                        if len(self.recent_winners) > self.window_size:
                            self.recent_winners.pop(0)  # Mantener solo últimos N
                        
                        # ✅ CORREGIDO: Solo valores numéricos en custom_metrics
                        # Convertir nombre del ganador a índice numérico
                        winner_idx = int(winner.split('_')[-1]) if '_' in winner else 0
                        episode.custom_metrics[f"tournament_winner_id"] = int(winner_idx)
                        episode.custom_metrics[f"tournament_completed"] = int(1)
                        
                        # Añadir resumen de porcentaje de acciones
                        if sum(self.current_tournament_actions.values()) > 0:
                            print(f"📊 Action Distribution:")
                            for agent in sorted(agents):
                                agent_actions = self.current_tournament_actions.get(agent, 0)
                                total_actions = sum(self.current_tournament_actions.values())
                                action_percentage = (agent_actions / total_actions) * 100 if total_actions > 0 else 0
                                print(f"   {agent}: {agent_actions} actions ({action_percentage:.1f}%)")
                    else:
                        episode.custom_metrics[f"tournament_winner_id"] = int(-1)  # Sin ganador
                        episode.custom_metrics[f"tournament_completed"] = int(1)
                    
                    self.tournament_count += 1
                    episode.custom_metrics["tournament_number"] = int(self.tournament_count)
                    
                    # Resetear contadores de acciones para el siguiente torneo
                    self.current_tournament_actions = {}
                
                # Añadir stacks individuales
                for agent in agents:
                    episode.custom_metrics[f"{agent}_final_stack"] = tournament.get_stack(agent)
                    episode.custom_metrics[f"{agent}_is_alive"] = 1 if tournament.get_stack(agent) > 0 else 0

        # ---- 5) Métricas de fase de juego ----
        if hasattr(raw_env, 'phase'):
            episode.custom_metrics["game_phase"] = raw_env.phase
            episode.custom_metrics["pot_size"] = getattr(raw_env, 'pot', 0)
            episode.custom_metrics["current_bet"] = getattr(raw_env, 'current_bet', 0)

# Configuración de APPO
N_PLAYERS = 4         

def env_creator(cfg):
    base_env = PokerTournamentEnv(n_agents=cfg.get("n_players", N_PLAYERS))
    return PettingZooEnv(base_env)       

# Registrar el entorno
from ray.tune.registry import register_env
register_env("poker_tournament", env_creator)



# Políticas separadas por agente
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared"

policies = {"shared": PolicySpec()}

config = (
    APPOConfig()
    .environment("poker_tournament", env_config={"n_players": N_PLAYERS})
    .framework("torch")
    .fault_tolerance(True)
    .training(
        lr=0.0001,
        clip_param=0.3,
        vtrace=True,
        use_critic=True,
        use_gae=True,
        lambda_=0.95,
        kl_coeff=0.5,
        entropy_coeff=0.01,
        vf_loss_coeff=0.5,
        model={
            "fcnet_hiddens": [256, 256],
            "use_lstm": True,
            "lstm_cell_size": 256,
            "max_seq_len": 20,
        },
        train_batch_size=4000,
        num_sgd_iter=10,
        grad_clip=40.0,
    )
    .resources(num_gpus=1)
    .rollouts(
        num_rollout_workers=8,
        rollout_fragment_length=200,
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn
    )
    .callbacks(PokerCallbacks)
)

# Configuración de WandB
wandb_cb = WandbLoggerCallback(
    project="PokerEntorn",
    group="APPO",
    log_config=True,
    upload_checkpoints=True
)

run_cfg = air.RunConfig(
    callbacks=[wandb_cb],
    stop={"training_iteration": 500},
    checkpoint_config=air.CheckpointConfig(
        checkpoint_frequency=20,
        checkpoint_at_end=True
    )
)

if __name__ == '__main__':
    ray.init(num_gpus=1)
    tuner = Tuner(
        "APPO",
        run_config=run_cfg,
        param_space=config.to_dict()
    )
    results = tuner.fit()
