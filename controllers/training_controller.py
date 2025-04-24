# Importações organizadas e simplificadas
import os
import logging
import random
from datetime import datetime
from typing import Tuple, List
import time
import torch.nn.functional as F


import numpy as np
import torch
import pygame
import matplotlib.pyplot as plt

from models.game.config import Config, RewardConfig
from models.utils.file_management import SaveManager, CheckpointManager
from models.agents.neural_networks import EnhancedSNN
from models.replay.buffer import PrioritizedReplayBuffer
from controllers.game_controller import TrashCollectionGame
from models.game.core import CompetitiveTrashGame

# Configuração de recompensas
reward_settings = {
    'TRASH_COLLECTED_REWARD': RewardConfig.Primary.TRASH_COLLECTED,
    'BATTERY_BONUS': RewardConfig.Battery.BONUS,
    'BATTERY_BONUS_MULTIPLIER': RewardConfig.Battery.BONUS_MULTIPLIER,
    'CHARGER_REWARD': RewardConfig.Battery.CHARGER_REWARD,
    'BATTERY_PENALTY': RewardConfig.Primary.BATTERY_PENALTY,
    'COMPLETION_BONUS': RewardConfig.Primary.COMPLETION_BONUS,
    'MOVEMENT_PENALTY': RewardConfig.Behavior.MOVEMENT_PENALTY,
    'IDLE_AT_CHARGER_PENALTY': RewardConfig.Battery.IDLE_AT_CHARGER,
    'IDLE_WITH_URGENT_TASKS_PENALTY': RewardConfig.Behavior.IDLE_WITH_URGENT_TASKS,
    'APPROACH_TRASH_REWARD': RewardConfig.Behavior.APPROACH_TRASH,
    'APPROACH_CHARGER_REWARD': RewardConfig.Behavior.APPROACH_CHARGER,
    'HIGH_BATTERY_PENALTY': RewardConfig.Battery.HIGH_BATTERY_PENALTY,
    'APPROACH_CHARGER_PENALTY': RewardConfig.Behavior.APPROACH_PENALTY_FACTOR,
    'AWAY_FROM_CHARGER_REWARD': RewardConfig.Behavior.AWAY_FROM_CHARGER,
    'AWAY_FROM_TRASH_PENALTY': RewardConfig.Avoidance.TRASH_BASE,
    'HIGH_BATTERY_AVOIDANCE_MULTIPLIER': RewardConfig.Avoidance.HIGH_BATTERY_MULTIPLIER,
    'CHARGER_HIGH_BATTERY': RewardConfig.ChargerThresholds.HIGH_BATTERY,
    'CHARGER_FULL_BATTERY': RewardConfig.ChargerThresholds.FULL_BATTERY,
    'APPROACH_PENALTY_FACTOR': RewardConfig.Behavior.APPROACH_PENALTY_FACTOR,
    'DISTANCE_REWARD_FACTOR': RewardConfig.Behavior.DISTANCE_REWARD_FACTOR,
    'APPROACH_REWARD_FACTOR': RewardConfig.Behavior.APPROACH_REWARD_FACTOR,
    'AWAY_FROM_TRASH_PENALTY_BASE': RewardConfig.Avoidance.TRASH_BASE,
    'AWAY_FROM_TRASH_MULTIPLIER_40': RewardConfig.Avoidance.MULTIPLIER_40,
    'AWAY_FROM_TRASH_MULTIPLIER_70': RewardConfig.Avoidance.MULTIPLIER_70,
}

for key, value in reward_settings.items():
    setattr(Config, key, value)

# Configuração do logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuração de diretórios
CHECKPOINT_DIR = "checkpoints"
BEST_MODELS_DIR = "saved_models/best_models"
TEMP_DIR = "temp"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

class MultiAgentTrainer:
    """Classe para treinamento de múltiplos agentes"""
    
    def __init__(self, num_agents: int = 1, with_curiosity: bool = True, render: bool = True): 
        """Inicializa o treinador
        
        Args:
            num_agents: Número de agentes
            with_curiosity: Flag para habilitar curiosidade
            render: Flag para habilitar renderização
        """
        self.save_manager = SaveManager()
        self.checkpoint_manager = CheckpointManager()
        self._try_load_previous_state()
        self.model_name = "trash_collector_ai"
        self.state_mean = None  # Média dos estados
        self.state_std = None   # Variância acumulada (depois vira desvio padrão)
        self.state_count = 0    # Número de estados observados
        # Tenta carregar modelo existente
        self._try_load_existing_model()
        self.num_agents = num_agents
        self.state_mean = None
        self.state_std = None
        self.state_count = 0
        self.games = [TrashCollectionGame(
            agent_color=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)),
            render=render) for _ in range(num_agents)]
            
        self.agent = EnhancedSNN(with_curiosity)
        self.target_network = EnhancedSNN(with_curiosity)
        self.target_network.load_state_dict(self.agent.state_dict())
        
        self.optimizer = torch.optim.AdamW(
            self.agent.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.9)
        
        self.replay_buffer = PrioritizedReplayBuffer()
        self.batch_size = Config.BATCH_SIZE
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.steps_done = 0
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self.with_curiosity = with_curiosity

    def update_epsilon(self):
        self.epsilon = max(Config.EPSILON_MIN, Config.EPSILON_START * (Config.EPSILON_DECAY ** (self.steps_done / 100)))

    def act(self, state: np.ndarray) -> int:
        """Seleciona uma ação usando política ε-greedy com normalização online.
        
        Args:
            state: Estado atual (não normalizado).
            
        Returns:
            Ação selecionada (0 a 4).
        """
        # 1. Normaliza o estado antes de qualquer processamento
        normalized_state = self.normalize_state(state)
        
        # 2. Exploração: ação aleatória (ε-greedy)
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        
        # 3. Exploração: usa a rede neural
        with torch.no_grad():
            # Converte o estado normalizado para tensor
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)
            
            # Obtém os Q-values (spikes) da rede
            spikes = self.agent(state_tensor)
            
            # Escolhe a ação com maior Q-value
            action = torch.argmax(spikes).item()
            
            # 4. Adiciona ruído aleatório à ação (10% de chance)
            if random.random() < 0.1:
                action = (action + random.choice([-1, 1])) % 5  # Garante ação válida (0-4)
            
            return action
        

    def visualize_all_agents_attention(self, save_dir: str = "attention_plots"):
        """Visualiza a atenção de todos os agentes simultaneamente"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Cria uma figura grande para todos os agentes
        plt.figure(figsize=(15, 5 * self.population_size))
        
        for i in range(self.population_size):
            # Obtém o estado do jogo para o agente
            state = self.game.get_state(i)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                # Processa através do modelo do agente específico
                visual_out = self.agents[i].visual_processing(state_tensor)
                attention = self.agents[i].spatial_attention(visual_out)
                visual_out = visual_out.detach().cpu().numpy().squeeze()
                attention = attention.detach().cpu().numpy().squeeze()
            
            # Gráfico da representação visual
            plt.subplot(self.population_size, 2, 2*i + 1)
            plt.bar(range(len(visual_out)), visual_out, color=self.game.agent_colors[i])
            plt.title(f"Agente {i} - Representação", fontsize=10)
            plt.xlabel("Neurônios")
            plt.ylabel("Ativação")
            plt.grid(True, linestyle='--', alpha=0.3)
            
            # Gráfico da atenção
            plt.subplot(self.population_size, 2, 2*i + 2)
            plt.bar(range(len(attention)), attention, color=self.game.agent_colors[i])
            plt.title(f"Agente {i} - Atenção", fontsize=10)
            plt.xlabel("Neurônios")
            plt.ylabel("Peso")
            plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Salva a figura
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"all_agents_attention_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de atenção salvo em: {save_path}")  # Print adicionado
        logger.info(f"Gráfico de todos os agentes salvo em {save_path}")
    def visualize_attention(self, agent_idx: int = 0, save_path: str = None):
        """Visualiza como a rede está processando o estado de um agente específico
        
        Args:
            agent_idx: Índice do agente a visualizar
            save_path: Caminho para salvar a imagem (opcional)
        """
        if not hasattr(self, 'games') or agent_idx >= len(self.games):
            print("Agente não encontrado")
            return
        
        # Obtém o estado atual do agente
        state = self.games[agent_idx].get_state()
        
        # Converte para tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Obtém ativações das camadas
        with torch.no_grad():
            visual_out = self.agent.visual_processing(state_tensor)
            attention = self.agent.spatial_attention(visual_out)
            
            # Converte para numpy
            visual_out = visual_out.detach().cpu().numpy().squeeze()
            attention = attention.detach().cpu().numpy().squeeze()
            
            # Cria figura
            plt.figure(figsize=(15, 6))
            
            # Gráfico da representação visual
            plt.subplot(1, 2, 1)
            plt.bar(range(len(visual_out)), visual_out, color='skyblue')
            plt.title(f"Representação Visual (Agente {agent_idx})", fontsize=12)
            plt.xlabel("Neurônios", fontsize=10)
            plt.ylabel("Ativação", fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Gráfico da atenção espacial
            plt.subplot(1, 2, 2)
            bars = plt.bar(range(len(attention)), attention, color='lightcoral')
            plt.title(f"Atenção Espacial (Agente {agent_idx})", fontsize=12)
            plt.xlabel("Neurônios", fontsize=10)
            plt.ylabel("Peso de Atenção", fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Destaca os 3 neurônios com maior atenção
            top_indices = np.argsort(attention)[-3:]
            for idx in top_indices:
                bars[idx].set_color('red')
            
            plt.tight_layout()
            
            # Salva ou mostra a figura
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Gráfico salvo em {save_path}")
            else:
                plt.show()
                
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        # Inicializa se for o primeiro estado
        if self.state_mean is None:
            self.state_mean = np.zeros_like(state)
            self.state_std = np.ones_like(state)  # Evita divisão por zero no início
        
        self.state_count += 1
        
        # Atualiza a média online
        delta = state - self.state_mean
        self.state_mean += delta / self.state_count
        
        # Atualiza a variância online (usando algoritmo de Welford)
        delta2 = state - self.state_mean
        self.state_std += delta * delta2
        
        # Calcula desvio padrão (com clipping para evitar valores extremos)
        std_dev = np.sqrt(self.state_std / self.state_count) + 1e-8
        
        # Normaliza e limita entre [-5, 5] para evitar outliers
        normalized_state = (state - self.state_mean) / std_dev
        return np.clip(normalized_state, -5, 5)
    
    def _try_load_previous_state(self):
        """Versão completamente segura para carregar estado anterior"""
        self.best_score = -float('inf')
        try:
            # 1. Tenta carregar o melhor modelo salvo
            loaded, metadata = self.save_manager.load_best_model(self.agent)
            
            if loaded:
                # Verificação EXTRA de segurança
                if not isinstance(metadata, dict):
                    metadata = {}
                    logger.warning("Metadados convertidos para dicionário vazio")
                
                logger.info(f"Modelo carregado. Score: {metadata.get('score', 'N/A')}")
                
                # Atribuições seguras com valores padrão
                self.best_score = float(metadata.get('score', -float('inf')))
                self.steps_done = int(metadata.get('steps_done', self.steps_done))
                self.epsilon = float(metadata.get('epsilon', self.epsilon))
                return

            # 2. Tenta carregar checkpoint se o modelo não foi carregado
            if self.checkpoint_manager.load_latest_checkpoint(self):
                logger.info("Checkpoint carregado com sucesso")
            else:
                logger.info("Iniciando treinamento do zero")

        except Exception as e:
            logger.error(f"Falha ao carregar estado anterior: {str(e)}")
            logger.info("Iniciando treinamento do zero")

    def save_model(self):
        try:
            metadata = {
                'steps_done': self.steps_done,
                'epsilon': self.epsilon,
                'save_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'performance': max(self.current_scores) if hasattr(self, 'current_scores') else 0
            }
            model_path = self.save_manager.save_model(self.agent, self.model_name, metadata)
            logger.info(f"Modelo salvo em: {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Erro ao salvar o modelo: {e}")

    def compute_intrinsic_reward(self, states: np.ndarray, actions: np.ndarray, 
                            next_states: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calcula recompensa intrínseca usando módulo de curiosidade
        
        Args:
            states: Estados atuais
            actions: Ações tomadas
            next_states: Próximos estados
            
        Returns:
            Tupla contendo:
            - Recompensas intrínsecas
            - Coeficiente de curiosidade
        """
        if not self.with_curiosity:
            return np.zeros(len(states)), 1.0

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        next_states_tensor = torch.FloatTensor(next_states)

        curiosity_reward = self.agent.curiosity(states_tensor, actions_tensor, next_states_tensor)

        self.agent.curiosity_optimizer.zero_grad()
        curiosity_loss = curiosity_reward.mean()
        curiosity_loss.backward()
        self.agent.curiosity_optimizer.step()

        # Curiosidade adaptativa
        self.agent.curiosity_coeff = max(0.1, 0.5 * (1 - self.steps_done / 1000))

        return curiosity_reward.detach().cpu().numpy(), self.agent.curiosity_coeff

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        
        """Normaliza o estado online"""
        if self.state_mean is None:
            self.state_mean = np.zeros_like(state)
            self.state_std = np.ones_like(state)
        
        # Atualiza estatísticas online
        self.state_count += 1
        delta = state - self.state_mean
        self.state_mean += delta / self.state_count
        delta2 = state - self.state_mean
        self.state_std += delta * delta2
        
        # Normaliza
        normalized = (state - self.state_mean) / (self.state_std + 1e-8)
        return np.clip(normalized, -5, 5)

    def replay(self) -> float:
        """Executa uma etapa de replay de experiência
        
        Returns:
            Valor da função de perda
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0

        samples = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, indices, weights = samples

        if self.with_curiosity:
            curiosity_rewards, curiosity_coeff = self.compute_intrinsic_reward(
                states, actions, next_states)
            rewards += curiosity_coeff * curiosity_rewards

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        weights_tensor = torch.FloatTensor(weights)

        # Calcula Q-values atuais
        current_q = self.agent(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        # Calcula Q-values alvo
        with torch.no_grad():
            next_actions = self.agent(next_states_tensor).argmax(1)
            next_q = self.target_network(next_states_tensor).gather(1, next_actions.unsqueeze(1))
            target = rewards_tensor.unsqueeze(1) + (1 - dones_tensor.unsqueeze(1)) * self.gamma * next_q

        # Atualiza prioridades no buffer
        errors = torch.abs(current_q - target).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, errors)

        # Calcula perda
        loss = (weights_tensor.unsqueeze(1) * F.mse_loss(current_q, target, reduction='none')).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip de gradientes para evitar explosão
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Atualiza rede alvo periodicamente
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.agent.state_dict())

        return loss.item()

    def train(self, episodes: int = 100) -> Tuple[List[List[float]], List[List[float]], List[List[int]]]:
        """Executa o loop principal de treinamento
        
        Args:
            episodes: Número de episódios para treinar
            
        Returns:
            Tupla contendo:
            - Histórico de recompensas por agente
            - Histórico de bateria por agente
            - Histórico de lixo coletado por agente
        """
        rewards_history = [[] for _ in range(self.num_agents)]
        battery_history = [[] for _ in range(self.num_agents)]
        trash_history = [[] for _ in range(self.num_agents)]

        try:
            for episode in range(episodes):
                states = [game.reset() for game in self.games]
                total_rewards = [0] * self.num_agents
                dones = [False] * self.num_agents

                while not all(dones):
                    for i, game in enumerate(self.games):
                        if dones[i]:
                            continue
                        
                        action = self.act(states[i])
                        next_state, reward, done = game.step(action)
                        dones[i] = done
                        states[i] = next_state
                        
                        # Renderiza SEMPRE, independente do estado
                        game.render(episode=episode, reward=reward)  # Atualiza a tela
                    self._process_episode_step(states, total_rewards, dones)
                    
                self._log_episode_results(episode, total_rewards, rewards_history, 
                                        battery_history, trash_history)
                
                # Salva checkpoint periódico
                if episode % 10 == 0:
                    self.checkpoint_manager.save_checkpoint(self, episode)
                
                # Avaliação e salvamento do melhor modelo
                current_max_score = self._evaluate_model()
                
                # Verifica se é um novo recorde
                if current_max_score > getattr(self, 'best_score', -float('inf')):
                    self.best_score = current_max_score
                    metadata = {
                        'episode': episode,
                        'score': current_max_score,
                        'epsilon': self.epsilon,
                        'steps_done': self.steps_done,
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    self.save_manager.save_best_model(self.agent, current_max_score, metadata)
                    logger.info(f"Novo recorde alcançado: {current_max_score:.2f}")

        except KeyboardInterrupt:
            logger.info("\nTreinamento interrompido pelo usuário")
        except Exception as e:
            logger.error(f"Erro durante o treinamento: {e}")
        finally:
            # Salva no final independente do que acontecer
            self._save_final_state()
            self.save_model()
            pygame.quit()
            return rewards_history, battery_history, trash_history
        
    def _evaluate_model(self) -> float:
        """Avalia o modelo atual e retorna a maior pontuação alcançada
        
        Returns:
            float: A maior pontuação (score) registrada pelo agente
        """
        # Verifica se temos scores registrados
        if not hasattr(self, 'current_scores') or not self.current_scores:
            return 0.0
        
        # Retorna o valor máximo encontrado
        max_score = max(self.current_scores)
        
        # Verificação adicional para garantir que é um número válido
        if not isinstance(max_score, (int, float)):
            logger.warning(f"Score inválido encontrado: {max_score}")
            return 0.0
        
        return float(max_score)
    def _save_final_state(self):
        """Salva estado final do treinamento"""
        self.checkpoint_manager.save_checkpoint(self)
        if hasattr(self, 'best_score'):
            self.save_manager.save_model(self.agent, "final_model")
            
            
    def _process_episode_step(self, states: List[np.ndarray], 
                            total_rewards: List[float], 
                            dones: List[bool]):
        """Processa um passo do episódio de treinamento"""
        actions = []  # Lista para armazenar todas as ações
        for i, game in enumerate(self.games):
            if dones[i]:
                game.freeze_agent()  # Congela o agente terminado
                continue
                
            self.steps_done += 1
            action = self.act(states[i])
            next_state, reward, done = game.step(action)
            
            self.replay_buffer.add(states[i], action, reward, next_state, done)
            total_rewards[i] += reward
            states[i] = next_state
            dones[i] = done
            
            loss = self.replay()
            self.update_epsilon()
            
            try:
                game.render( episode=0 , reward=total_rewards[i])
            except Exception as e:
                logger.warning(f"Erro ao renderizar: {e}")
            
            if any(event.type == pygame.QUIT for event in pygame.event.get()):
                pygame.quit()
                return

    def _log_episode_results(self, episode: int, 
                        total_rewards: List[float],
                        rewards_history: List[List[float]], 
                        battery_history: List[List[float]], 
                        trash_history: List[List[int]]):
        """Registra os resultados do episódio"""
        avg_reward = sum(total_rewards) / self.num_agents
        avg_battery = sum(game.battery for game in self.games) / self.num_agents
        avg_trash = sum(game.collected_trash for game in self.games) / self.num_agents
        
        logger.info(f"Episódio {episode + 1}:")
        logger.info(f"  Recompensa Média: {avg_reward:.2f}")
        logger.info(f"  Bateria Média: {avg_battery:.1f}%")
        logger.info(f"  Lixo Coletado: {avg_trash}/10")
        logger.info(f"  Exploração (ε): {self.epsilon:.3f}")
        logger.info(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        for i in range(self.num_agents):
            rewards_history[i].append(total_rewards[i])
            battery_history[i].append(self.games[i].battery)
            trash_history[i].append(self.games[i].collected_trash)

class EvolutionaryTrainer:
    """Treinador evolucionário para múltiplos agentes competitivos"""

    def __init__(self, population_size: int = 5, with_curiosity: bool = True, 
                render: bool = True):
        """Inicializa o treinador evolucionário
        
        Args:
            population_size: Tamanho da população (deve ser ≥ 2)
            with_curiosity: Flag para habilitar curiosidade
            render: Flag para habilitar renderização
        """
        # 1. Validação de parâmetros
        if population_size < 2:
            raise ValueError("Population size deve ser pelo menos 2")
        
        # 2. Configuração básica
        self.population_size = population_size
        self.with_curiosity = with_curiosity
        self.model_name = "evolutionary_trash_collector"
        self.selection_interval = 5  # Número de episódios entre seleções
        self.generation = 0
        self.start_time = time.time()
        self.time_limit = Config.TIME_LIMIT
        
        # 3. Inicialização de componentes principais
        self.save_manager = SaveManager()
        
        # 4. Inicialização dos agentes
        self.agents = [EnhancedSNN(with_curiosity) for _ in range(population_size)]
        self.optimizers = [
            torch.optim.AdamW(agent.parameters(), lr=Config.LEARNING_RATE) 
            for agent in self.agents
        ]
        
        # 5. Configuração do ambiente de jogo
        self.game = CompetitiveTrashGame(population_size, render)
        
        # 6. Estado do treinamento
        self.generation_best_scores = [0.0] * self.population_size
        self.current_scores = [0] * population_size
        self.best_score = -float('inf')
        self.best_agent_idx = -float('inf')
        self.best_agent = None  # Inicializa como None
        self.last_improvement = 0
        self.generation_scores = [[] for _ in range(self.population_size)]

        # 7. Carregar modelo existente (se houver)
        self._load_best_model()
        
        logger.info(f"EvolutionaryTrainer inicializado com {population_size} agentes")
        
    def _load_best_model(self):
        """Carrega o melhor modelo salvo para todos os agentes"""
        loaded, metadata = self.save_manager.load_best_model(self.agents[0])
        if loaded:
            # Verifica se metadata é um dicionário antes de acessar
            if isinstance(metadata, dict):
                logger.info(f"Carregado melhor modelo com score: {metadata.get('score', 'N/A')}")
                # Copia os pesos para todos os agentes
                for i in range(1, self.population_size):
                    self.agents[i].load_state_dict(self.agents[0].state_dict())
                self.best_score = metadata.get('score', -float('inf'))
            else:
                logger.warning("Metadados não estão no formato esperado (dicionário)")

    def save_best_model(self):
        """Salva o melhor modelo da população atual se for um novo recorde
        
        Processo:
        1. Avalia todos os agentes e identifica o melhor da geração atual
        2. Compara com o melhor score histórico
        3. Salva apenas se for um novo recorde
        4. Gera logs e mensagens informativas
        """
        # 1. Identifica o melhor agente atual
        ranked_agents = self.evaluate_agents()
        current_best_idx, current_best_score = ranked_agents[0]
        
        # Log detalhado do ranking
        logger.info("=== Ranking de Agentes (por melhor desempenho individual) ===")
        for rank, (agent_idx, score) in enumerate(ranked_agents[:5], 1):  # Mostra top 5
            logger.info(f"{rank}º - Agente {agent_idx}: {score:.2f} pts")
        
        
        metadata = {
            'generation': self.generation,
            'score': current_best_score,
            'agent_index': current_best_idx,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'selection_criteria': 'best_individual_performance'
        }
        
        # Nome do arquivo inclui geração e score
        filename = f"gen_{self.generation}_score_{current_best_score:.2f}_{metadata['timestamp']}.pt"
        model_path = os.path.join(self.save_manager.best_model_dir, filename)
        
        
        
        torch.save({
            'model_state_dict': self.agents[current_best_idx].state_dict(),
            'metadata': metadata
        }, model_path)
        
        logger.info(f"Melhor modelo da geração {self.generation} salvo: {model_path}")
        # 2. Inicializa best_score se for a primeira execução
        if not hasattr(self, 'best_score'):
            self.best_score = -float('inf')
            self.best_agent_idx = -float('inf')
            
            
        if current_best_score > self.best_score:
            self.best_score = current_best_score
            self.best_agent_idx = current_best_idx
            
            # Log do novo recorde
            logger.info(f"\n NOVO RECORDE - Agente {current_best_idx} selecionado!")
            logger.info(f" Melhor desempenho individual: {current_best_score:.2f} pts")
            logger.info(f" Salvando como novo melhor modelo...")
            
            metadata = {
                'generation': self.generation,
                'score': current_best_score,
                'agent_index': current_best_idx,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'selection_criteria': 'best_individual_performance'
            }
            
            self.save_manager.save_best_model(
                self.agents[current_best_idx], 
                current_best_score, 
                metadata
            )
        else:
            # Log quando não há novo recorde
            logger.info(f"\n Melhor agente da geração: Agente {current_best_idx}")
            logger.info(f" Desempenho: {current_best_score:.2f} pts")
            logger.info(f" Recorde atual: {self.best_score:.2f} pts (Agente {self.best_agent_idx})")
        
        # 3. Verifica se é um novo recorde
        if current_best_score > self.best_score:
            self._save_new_record(current_best_idx, current_best_score)
        else:
            self._log_no_improvement(current_best_score)

    def _save_new_record(self, agent_idx: int, score: float):
        """Salva um novo modelo recorde e atualiza os parâmetros"""
        self.best_score = score
        self.best_agent_idx = agent_idx
        
        metadata = {
            'generation': self.generation,
            'score': score,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        model_path = self.save_manager.save_best_model(
            self.agents[agent_idx], 
            score, 
            metadata
        )
        
        # Logging e feedback visual
        logger.info(f"Novo recorde! Score: {score:.2f}")
        print(f"✅ Novo modelo salvo como recorde em: {model_path}")

    def _log_no_improvement(self, current_score: float):
        """Registra quando não há melhoria no score"""
        logger.info(
            f"Melhor score desta geração: {current_score:.2f} "
            f"(Não superou o recorde: {self.best_score:.2f})"
        )
        print(
            f"⏭ Melhor score da geração {current_score:.2f} "
            f"não superou o recorde {self.best_score:.2f}"
        )
            
    def save_models(self):
        """Salva todos os modelos da população com numeração consistente"""
        for i, agent in enumerate(self.agents):
            metadata = {
                'generation': self.generation,
                'save_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'performance': self.game.current_scores[i],
                'agent_index': i  # Adiciona o índice do agente
            }
            # Adiciona o índice do agente no nome do arquivo
            model_path = self.save_manager.save_model(
                agent, 
                f"{self.model_name}_agent_{i}", 
                metadata
            )

    def evaluate_agents(self) -> List[Tuple[int, float]]:
        """Avaliação com verificações de segurança"""
        try:
            # 1. Verifica se temos dados para avaliar
            if not hasattr(self, 'generation_scores') or not self.generation_scores:
                logger.warning("Nenhum dado de geração disponível para avaliação")
                return [(i, 0.0) for i in range(self.population_size)]

            # 2. Verifica consistência dos dados
            if len(self.generation_scores) != self.population_size:
                logger.error(f"Tamanho inconsistente: generation_scores ({len(self.generation_scores)}) != population_size ({self.population_size})")
                return [(i, 0.0) for i in range(self.population_size)]

            # 3. Calcula os scores máximos de cada agente
            ranked = []
            for i in range(self.population_size):
                if not self.generation_scores[i]:  # Se lista vazia
                    logger.warning(f"Agente {i} sem scores registrados")
                    ranked.append((i, 0.0))
                else:
                    try:
                        max_score = max(self.generation_scores[i])
                        ranked.append((i, float(max_score)))
                    except Exception as e:
                        logger.error(f"Erro ao calcular max_score para agente {i}: {str(e)}")
                        ranked.append((i, 0.0))

            # 4. Ordena do melhor para o pior
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked

        except Exception as e:
            logger.error(f"Erro crítico ao avaliar agentes: {str(e)}")
            # Retorna uma lista padrão em caso de erro crítico
            return [(i, 0.0) for i in range(self.population_size)]

    def select_and_reproduce(self) -> float:
        """Seleciona os melhores agentes e realiza reprodução com mutação.
        
        Returns:
            float: Tempo decorrido no processo de seleção/reprodução
        """
        # 1. Pré-validações e preparação
        start_time = time.time()
        
        if not hasattr(self, 'best_score'):
            self.best_score = -float('inf')

        # 2. Avaliação e ranking dos agentes
        try:
            ranked_agents = self._get_ranked_agents()
            best_agent_idx, best_score = ranked_agents[0]
        except Exception as e:
            logger.error(f"Falha na avaliação: {str(e)}")
            raise RuntimeError("Falha no processo de seleção") from e
        if best_score > self.best_score:
            self.best_score = best_score
            self.best_agent_idx = best_agent_idx
            # Salvar o modelo apenas se for um novo recorde
            self._save_best_model(best_agent_idx, best_score)

        # 3. Atualização de recordes
        record_updated = self._update_records(best_agent_idx, best_score, ranked_agents)

        # 4. Limpeza de modelos antigos
        self._cleanup_unused_models()

        # 5. Processo de reprodução e mutação
        self._reproduce_agents(best_agent_idx)

        # 6. Finalização
        self.generation += 1
        self.game.reset()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Seleção concluída - Geração {self.generation} | Tempo: {elapsed_time:.2f}s")
        return elapsed_time

    # --- Métodos auxiliares ---

    def _get_ranked_agents(self) -> List[Tuple[int, float]]:
        """Retorna agentes ordenados por desempenho com validação."""
        if not hasattr(self.game, 'current_scores'):
            raise AttributeError("current_scores não encontrado no game")
        
        scores = self.game.current_scores
        if len(scores) != self.population_size:
            raise ValueError(f"Tamanho incorreto de current_scores: {len(scores)} != {self.population_size}")

        ranked = sorted(
            [(i, float(score)) for i, score in enumerate(scores)],
            key=lambda x: x[1], 
            reverse=True
        )
        
        logger.debug(f"Ranking calculado: {ranked}")
        self._log_ranking(ranked)
        return ranked

    def _update_records(self, best_idx: int, best_score: float, 
                    ranked_agents: List[Tuple[int, float]]) -> bool:
        """Atualiza os recordes e retorna se houve melhoria."""
        record_updated = False
        
        if best_score > self.best_score:
            improvement = best_score - self.best_score
            self.best_score = best_score
            self.best_agent_idx = best_idx
            record_updated = True
            
            logger.info(f"Novo recorde! Score: {best_score:.2f} (+{improvement:.2f})")
            print(f"\n{'='*50}")
            print(f"🔥 NOVO RECORDE - Agente {best_idx}: {best_score:.2f} pts")
            print(f"Geração: {self.generation} | Melhoria: +{improvement:.2f}")
            print(f"{'='*50}\n")
        else:
            logger.info(f"Melhor score da geração: {best_score:.2f} (Recorde: {self.best_score:.2f})")

        return record_updated

    def _log_ranking(self, ranked_agents: List[Tuple[int, float]]):
        """Log detalhado do ranking de agentes (apenas top 5)"""
        log_msg = ["Top 5 Agentes da Geração:"]
        for rank, (agent_idx, score) in enumerate(ranked_agents[:5], 1):
            log_msg.append(f"{rank}º - Agente {agent_idx}: {score:.2f} pts (Melhor da geração)")
        
    def _cleanup_unused_models(self):
        """Remove modelos de agentes não utilizados."""
        try:
            current_agents = list(range(self.population_size))
            self.save_manager.cleanup_unused_models(current_agents)
        except Exception as e:
            logger.error(f"Erro na limpeza de modelos: {str(e)}")

    def _reproduce_agents(self, best_agent_idx: int):
        """Realiza a reprodução e mutação dos agentes."""
        # Copia o melhor agente para os demais
        for i in range(self.population_size):
            if i != best_agent_idx:
                try:
                    self.agents[i].load_state_dict(self.agents[best_agent_idx].state_dict())
                    
                    # Aplica mutação
                    for param in self.agents[i].parameters():
                        noise = 0.05 * torch.randn_like(param.data)
                        param.data += noise * (1 - (i/self.population_size))  # Mutação decrescente
                except Exception as e:
                    logger.error(f"Erro ao reproduzir agente {i}: {str(e)}")
                    raise

    def train(self, total_episodes: int = 500, time_limit: float = None) -> List[float]:
        """Executa o treinamento evolucionário
        
        Args:
            total_episodes: Número total de episódios de treinamento
            time_limit: Tempo máximo por episódio em segundos (None para usar o padrão)
            
        Returns:
            List[float]: Histórico das recompensas máximas por episódio
        """
        # 1. Inicialização
        rewards_history = []
        time_limit = self.time_limit if time_limit is None else float(time_limit)
        self._initialize_training_state()
        
        try:
            # 2. Loop principal de treinamento
            for episode in range(total_episodes):
                episode_reward = self._run_episode(episode, total_episodes, time_limit)
                rewards_history.append(episode_reward)
                
                # 3. Processamento pós-episódio
                self._post_episode_processing(episode)
                
        except KeyboardInterrupt:
            logger.info("Treinamento interrompido pelo usuário")
        except Exception as e:
            logger.error(f"Erro durante o treinamento: {str(e)}")
            raise
        finally:
            # 4. Finalização garantida
            return self._finalize_training(rewards_history)

    # --- Métodos auxiliares ---

    def _initialize_training_state(self):
        """Prepara o estado inicial do treinamento"""
        if not hasattr(self, 'best_score'):
            self.best_score = -float('inf')
            self.best_agent_idx = -float('inf')
            self.last_improvement = -float('inf')
        logger.info(f"Iniciando treinamento com {self.population_size} agentes")

    def _run_episode(self, episode: int, total_episodes: int, time_limit: float) -> float:
        """Executa um único episódio de treinamento"""
        episode_start = time.time()
        logger.info(f"\n=== Episódio {episode + 1}/{total_episodes} ===")
        
        # 1. Reset do ambiente
        states = self.game.reset()
        dones = [False] * self.population_size
        self.episode_scores = [0] * self.population_size  # Scores do episódio atual
        episode_max_score = -9999  # Melhor score individual do episódio

        # 2. Loop do episódio
        while not all(dones):
            elapsed = time.time() - episode_start
            if elapsed > time_limit:
                self._handle_timeout(dones)
                break

            # 3. Coleta de ações
            actions = self._collect_actions(states, episode, total_episodes)
            
            # 4. Execução no ambiente
            next_states, rewards, dones = self.game.step(actions)
            
            # 5. Atualização de estados
            self._update_training_state(states, actions, rewards, next_states, dones)
            states = next_states

            # 6. Renderização
            self.game.render(episode, max(self.episode_scores), elapsed)
            
            # Atualiza scores
            current_max = max(rewards) if rewards else 0
            if current_max > episode_max_score:
                episode_max_score = current_max
            
            # Atualiza ambos os sistemas de pontuação
            for i, r in enumerate(rewards):
                self.episode_scores[i] += r
                self.game.current_scores[i] += r  # Garante que o jogo também atualize
            
        # 7. Processamento final
        max_score = max(self.episode_scores)
        logger.info(f"Episódio {episode + 1} concluído | Score máximo: {max_score:.2f}")
        return max(self.episode_scores), episode_max_score

    def _collect_actions(self, states: List[np.ndarray], episode: int, total_episodes: int) -> List[int]:
        """Coleta ações para todos os agentes usando política ε-greedy"""
        actions = []
        epsilon = max(0.1, 0.5 * (1 - episode / total_episodes))  # Decaimento da exploração
        
        for i, state in enumerate(states):
            if self.game.frozen_states.get(i, False):
                actions.append(0)  # Ação nula se congelado
                continue
                
            if random.random() < epsilon:
                actions.append(random.randint(0, 4))  # Exploração
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state)
                    actions.append(torch.argmax(self.agents[i](state_tensor)).item())
        
        return actions

    def _update_training_state(self, states: List[np.ndarray], rewards: List[float], 
                            next_states: List[np.ndarray], dones: List[bool], actions):
        """Atualiza o estado do treinamento"""
        for i in range(self.population_size):
            self.episode_scores[i] += rewards[i]
            
            # Atualiza memória/replay buffer se necessário
            if hasattr(self, 'replay_buffer'):
                self.replay_buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def _handle_timeout(self, dones: List[bool]):
        """Trata casos de timeout do episódio"""
        for i in range(self.population_size):
            if not dones[i]:
                reward, _ = self.game._calculate_rewards_and_done(i)
                self.episode_scores[i] += reward
        
        logger.warning(f"Timeout alcançado | Scores: {self.episode_scores}")

    def _post_episode_processing(self, episode: int):
        """Processamentos após cada episódio"""
        
        # Atualiza o melhor score da geração para cada agente
        for i in range(self.population_size):
            if self.episode_scores[i] > self.generation_best_scores[i]:
                self.generation_best_scores[i] = self.episode_scores[i]
                logger.debug(f"Agente {i} alcançou novo melhor pessoal: {self.episode_scores[i]:.2f} pts")
                
                
        # 1. Registrar scores da geração atual
        for i in range(self.population_size):
            self.generation_scores[i].append(self.episode_scores[i])
        
        # 2. Seleção periódica a cada selection_interval episódios
        if (episode + 1) % self.selection_interval == 0:
            ranked_agents = self.evaluate_agents()

            try:
                
                for i in range(self.population_size):
                    if self.generation_scores[i]:  # Verifica se há scores
                        max_score = max(self.generation_scores[i])
                        logger.info(f"Agente {i}: Máximo {max_score:.2f} pts | Média {np.mean(self.generation_scores[i]):.2f} pts")
                
                best_gen_score = max(max(scores) for scores in self.generation_scores)
                
                # Salva o melhor da geração independente de ser recorde
                self.save_best_model()
                
                # Depois faz a seleção e reprodução
                elapsed = self.select_and_reproduce()
                
                logger.info(f"Seleção realizada em {elapsed:.2f}s | "
                        f"Geração {self.generation} | "
                        f"Melhor score da geração: {best_gen_score:.2f}")
                
                
                self.generation_scores = [[] for _ in range(self.population_size)]
                self.generation += 1  # Incrementa a geração
                
                
            except Exception as e:
                logger.error(f"Erro na seleção: {str(e)}")
        
        # 2. Log de progresso
        best_agent_idx = np.argmax(self.episode_scores)
        current_max = self.episode_scores[best_agent_idx]

        if current_max > self.best_score:
            logger.info(f"\n NOVO RECORDE INDIVIDUAL! Agente {best_agent_idx} alcançou {current_max:.2f} pts")
            logger.info(f" Superou o recorde anterior de {self.best_score:.2f} pts (Agente {self.best_agent_idx})")
        else:
            logger.info(f"\n Melhor desempenho do episódio: Agente {best_agent_idx} com {current_max:.2f} pts")
            logger.info(f" Recorde mantido: {self.best_score:.2f} pts (Agente {self.best_agent_idx})")

        # Adicional: mostra os top 3 agentes do episódio
        sorted_scores = sorted([(i, score) for i, score in enumerate(self.episode_scores)], 
                            key=lambda x: x[1], reverse=True)
        logger.info(" Top 3 deste episódio:")
        for rank, (agent_idx, score) in enumerate(sorted_scores[:3], 1):
            logger.info(f"{rank}º - Agente {agent_idx}: {score:.2f} pts")

    def _finalize_training(self, rewards_history: List[float]) -> List[float]:
        """Finalização garantida do treinamento"""
        try:
            # 1. Salvamento final
            self.save_manager.cleanup_unused_models([self.best_agent_idx])
            self.save_best_model()
            self.save_models()
            
            # 2. Relatório final
            if rewards_history:
                logger.info(f"Treinamento concluído | Melhor score: {max(rewards_history):.2f}")
                print(f"\nTreinamento finalizado! Melhor score alcançado: {max(rewards_history):.2f}")
            
            return rewards_history
        except Exception as e:
            logger.error(f"Erro na finalização: {str(e)}")
            return rewards_history