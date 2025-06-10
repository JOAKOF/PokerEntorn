# ♠️ Entorn Multiagent de Texas Hold'em 4-Max per a Aprenentatge per Reforç

Aquest repositori conté el desenvolupament complet del meu Treball de Fi de Grau (TFG), que té com a objectiu crear un entorn obert, configurable i multiagent per entrenar agents d'aprenentatge per reforç en un context de pòquer amb informació incompleta.

## 🎯 Objectius del projecte

- Implementar un entorn de **Texas Hold'em Limit 4-Max** compatible amb la interfície estàndard de **PettingZoo**.
- Permetre entrenaments amb algoritmes de **Reinforcement Learning multiagent (MARL)** com *Independent DQN* i *APPO*.
- Registrar mètriques avançades com winrates, accions per agent, distribució de fitxes i fases del joc.
- Avaluar quins mètodes s’apropen més a estratègies properes a l’equilibri de joc.

## 🏗️ Arquitectura i components

- `poker_env.py`: Entorn principal del joc amb compatibilitat per a PettingZoo.
- `deck.py` i `tournament.py`: Mòduls per gestionar la baralla i la lògica del torneig.
- `train_iql.py`: Entrenament amb Independent Q-learning (DQN per agent).
- `train_appo.py`: Entrenament amb APPO i xarxes LSTM compartides.
- `compare_algorithms.py`: Script per comparar resultats entre diferents agents.
- `requirements.txt`: Dependències del projecte.

## 📊 Entorn i mètriques

L’entorn simula tornejos complets amb rotació de blinds, recompenses per mà i recompenses acumulades. A través de callbacks personalitzats es recullen mètriques com:

- Winrate per agent (total i recent)
- Accions per mà i percentatge per agent
- Distribució de fitxes i eliminacions
- Fase del joc, mida del pot, aposta actual

## 🧠 Algoritmes provats

S’han implementat i entrenat dues aproximacions:

- `DQN`: polítiques independents, buffer amb priorització, epsilon-greedy lent.
- `APPO`: política compartida amb LSTM, estimació d’avantatges amb GAE i V-Trace.

> Tots els hiperparàmetres es poden consultar i modificar dins dels arxius `train_iql.py` i `train_appo.py`.

## 📎 Treball de Fi de Grau

Pots consultar el document complet del TFG aquí:  
📄 [Descarrega el PDF](./TFG_Joaquin_Flores.pdf)

