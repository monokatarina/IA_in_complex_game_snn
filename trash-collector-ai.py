import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import os
import math
import time
import logging
from typing import List, Tuple, Dict, Optional, Any
import json
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== CONSTANTS AND CONFIGURATION ====================
class Config:
    """Configurações globais do jogo e treinamento"""
    
    # Configurações do jogo
    GAME_WIDTH = 800
    GAME_HEIGHT = 600

    BATTERY_DRAIN_MOVE = 0.2
    BATTERY_DRAIN_IDLE = 0.1
    TRASH_COLLECTION_RADIUS = 20
    CHARGER_RADIUS = 25
    MAX_BATTERY = 100
    MIN_TRASH_DISTANCE = 45  # Distância mínima entre objetos
    SECTOR_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    TIME_LIMIT = 300 # Limite de tempo para cada episódio (em segundos)
    
    # Configurações da rede neural
    MEMORY_SIZE = 10 
    STATE_SIZE = 22 + MEMORY_SIZE * 4 
    
    # Configurações de treinamento
    BATCH_SIZE = 64           # Número de amostras por batch               
    GAMMA = 0.95               # Fator de desconto para recompensas futuras               
    EPSILON_START = 1.0        # Valor inicial de epsilon           
    EPSILON_MIN = 0.05         # Epsilon mínimo para exploração         
    EPSILON_DECAY = 0.995      # Taxa de decaimento de epsilon     
    TARGET_UPDATE_FREQ = 300   # Frequência de atualização da rede alvo    
    LEARNING_RATE = 0.0005     # Taxa de aprendizado da rede neural   
    
    #config arquivo
    os.makedirs("saved_models/best_models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
# ==================== REWARD CONFIGURATION ====================
class RewardConfig:
    """Configurações hierárquicas de recompensa para o agente"""
    
    # Recompensas principais
    class Primary:
        TRASH_COLLECTED = 400.0       # Recompensa por coletar lixo
        COMPLETION_BONUS = 250.0      # Bônus por coletar todo o lixo
        BATTERY_PENALTY = -100.0       # Penalidade por bateria zerada
    
    # Bateria e carregador
    class Battery:
        BONUS = 15.0                  # Bônus base por nível de bateria
        BONUS_MULTIPLIER = 1.7        # Multiplicador quando bateria >70%
        CHARGER_REWARD = 1.0          # Recompensa base por usar carregador
        
        # Limiares
        HIGH_LEVEL = 0.6              # Bateria considerada alta (>70%)
        LOW_LEVEL = 0.3               # Bateria considerada baixa (<30%)
        CRITICAL_LEVEL = 0.1          # Bateria considerada crítica (<10%)
        
        # Penalidades
        IDLE_AT_CHARGER = -3.0        # Ficar parado no carregador com bateria alta
        HIGH_BATTERY_PENALTY = -2.0   # Penalidade por bateria alta em situações inadequadas
    
    # Comportamento
    class Behavior:
        MOVEMENT_PENALTY = -0.05      # Penalidade por se mover (incentivo à eficiência)
        IDLE_WITH_URGENT_TASKS = -2.5 # Ficar parado com tarefas urgentes
        
        # Fatores de aproximação
        APPROACH_TRASH = 1.0          # Recompensa por se aproximar do lixo
        APPROACH_CHARGER = 2.0        # Recompensa por se aproximar do carregador
        AWAY_FROM_CHARGER = 3.0       # Recompensa por se afastar do carregador (quando bateria cheia)
        
        # Fatores de distância
        APPROACH_PENALTY_FACTOR = 0.3 # Penalidade por aproximação desnecessária
        DISTANCE_REWARD_FACTOR = 0.2  # Recompensa por manter distância adequada
        APPROACH_REWARD_FACTOR = 0.5  # Fator de recompensa por aproximação
    
    # Penalidades por afastamento
    class Avoidance:
        TRASH_BASE = -1             # Penalidade base por se afastar do lixo
        MULTIPLIER_40 = 1.5           # Multiplicador quando bateria >40%
        MULTIPLIER_70 = 2.0           # Multiplicador quando bateria >70%
        HIGH_BATTERY_MULTIPLIER = 2.0 # Multiplicador adicional para bateria alta
    
    # Limiares do carregador
    class ChargerThresholds:
        HIGH_BATTERY = 0.5            # Limiar para considerar bateria alta no carregador
        FULL_BATTERY = 0.15            # Limiar para considerar bateria cheia

# Atribuição para manter compatibilidade
Config.TRASH_COLLECTED_REWARD = RewardConfig.Primary.TRASH_COLLECTED
Config.BATTERY_BONUS = RewardConfig.Battery.BONUS
Config.BATTERY_BONUS_MULTIPLIER = RewardConfig.Battery.BONUS_MULTIPLIER
Config.CHARGER_REWARD = RewardConfig.Battery.CHARGER_REWARD
Config.BATTERY_PENALTY = RewardConfig.Primary.BATTERY_PENALTY
Config.COMPLETION_BONUS = RewardConfig.Primary.COMPLETION_BONUS
Config.MOVEMENT_PENALTY = RewardConfig.Behavior.MOVEMENT_PENALTY
Config.IDLE_AT_CHARGER_PENALTY = RewardConfig.Battery.IDLE_AT_CHARGER
Config.IDLE_WITH_URGENT_TASKS_PENALTY = RewardConfig.Behavior.IDLE_WITH_URGENT_TASKS
Config.APPROACH_TRASH_REWARD = RewardConfig.Behavior.APPROACH_TRASH
Config.APPROACH_CHARGER_REWARD = RewardConfig.Behavior.APPROACH_CHARGER
Config.HIGH_BATTERY_PENALTY = RewardConfig.Battery.HIGH_BATTERY_PENALTY
Config.APPROACH_CHARGER_PENALTY = RewardConfig.Behavior.APPROACH_PENALTY_FACTOR
Config.AWAY_FROM_CHARGER_REWARD = RewardConfig.Behavior.AWAY_FROM_CHARGER
Config.AWAY_FROM_TRASH_PENALTY = RewardConfig.Avoidance.TRASH_BASE
Config.HIGH_BATTERY_AVOIDANCE_MULTIPLIER = RewardConfig.Avoidance.HIGH_BATTERY_MULTIPLIER
Config.CHARGER_HIGH_BATTERY = RewardConfig.ChargerThresholds.HIGH_BATTERY
Config.CHARGER_FULL_BATTERY = RewardConfig.ChargerThresholds.FULL_BATTERY
Config.APPROACH_PENALTY_FACTOR = RewardConfig.Behavior.APPROACH_PENALTY_FACTOR
Config.DISTANCE_REWARD_FACTOR = RewardConfig.Behavior.DISTANCE_REWARD_FACTOR
Config.APPROACH_REWARD_FACTOR = RewardConfig.Behavior.APPROACH_REWARD_FACTOR
Config.AWAY_FROM_TRASH_PENALTY_BASE = RewardConfig.Avoidance.TRASH_BASE
Config.AWAY_FROM_TRASH_MULTIPLIER_40 = RewardConfig.Avoidance.MULTIPLIER_40
Config.AWAY_FROM_TRASH_MULTIPLIER_70 = RewardConfig.Avoidance.MULTIPLIER_70
# ==================== class save ====================
class SaveManager:
    """Gerencia o salvamento e carregamento de modelos IA com segurança"""
    
    def __init__(self, save_dir="saved_models"):
        self.save_dir = save_dir
        self.best_model_dir = os.path.join(save_dir, "best_models")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        self._setup_safe_loading()

    def _setup_safe_loading(self):
        """Configura globais seguros para torch.load"""
        try:
            from numpy.core.multiarray import scalar
            import torch.serialization
            torch.serialization.add_safe_globals([scalar])
        except ImportError as e:
            logger.warning(f"Não foi possível configurar carregamento seguro: {e}")

    def _get_timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_model(self, model, model_name="model", metadata=None):
        """Salva o modelo com metadados"""
        timestamp = self._get_timestamp()
        model_path = os.path.join(self.save_dir, f"{model_name}_{timestamp}.pt")
        metadata_path = os.path.join(self.save_dir, f"{model_name}_{timestamp}.json")

        # Garante metadados mínimos
        default_metadata = {
            'save_date': timestamp,
            'model_name': model_name,
            'framework': 'pytorch'
        }
        if metadata:
            default_metadata.update(metadata)

        # Salva de forma segura
        torch.save(model.state_dict(), model_path)
        with open(metadata_path, 'w') as f:
            json.dump(default_metadata, f, indent=4)

        return model_path

    def load_model(self, model, model_path):
        """Carrega um modelo específico com verificação de segurança"""
        try:
            model.load_state_dict(torch.load(model_path, weights_only=True))
            return True
        except Exception as e:
            logger.error(f"Falha ao carregar modelo: {e}")
            return False

    def save_best_model(self, model, score, metadata=None):
        """Salva o modelo apenas se for melhor que os existentes"""
        metadata = metadata or {}
        if not isinstance(metadata, dict):
            metadata = {}
        
        metadata.update({
            'score': float(score),
            'save_date': self._get_timestamp(),
            'is_best': True
        })

        filename = f"best_model_score_{score:.2f}_{self._get_timestamp()}.pt"
        model_path = os.path.join(self.best_model_dir, filename)

        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }, model_path)

        self._cleanup_old_models()
        return model_path

    def load_best_model(self, model):
        """Versão robusta que sempre retorna metadados como dicionário"""
        try:
            model_files = [f for f in os.listdir(self.best_model_dir) 
                        if f.startswith('best_model_') and f.endswith('.pt')]
            
            if not model_files:
                return False, {}  # Retorna dicionário vazio

            best_model = None
            best_score = -float('inf')
            best_metadata = {}

            for model_file in model_files:
                try:
                    model_path = os.path.join(self.best_model_dir, model_file)
                    data = torch.load(model_path, weights_only=True, map_location='cpu')
                    
                    # Garante que temos metadados no formato correto
                    if not isinstance(data, dict):
                        continue
                        
                    file_metadata = data.get('metadata', {})
                    if not isinstance(file_metadata, dict):
                        file_metadata = {}
                    
                    file_score = float(file_metadata.get('score', -float('inf')))
                    
                    if file_score > best_score:
                        best_score = file_score
                        best_model = model_path
                        best_metadata = file_metadata
                        
                except Exception as e:
                    logger.warning(f"Erro ao processar {model_file}: {str(e)}")
                    continue

            if best_model:
                data = torch.load(best_model, weights_only=True, map_location='cpu')
                model.load_state_dict(data['model_state_dict'])
                return True, best_metadata

            return False, {}
            
        except Exception as e:
            logger.error(f"Erro crítico ao carregar modelo: {str(e)}")
            return False, {}

    def _cleanup_old_models(self, keep=3):
        """Mantém apenas os 'keep' melhores modelos"""
        model_files = [f for f in os.listdir(self.best_model_dir) 
                    if f.startswith('best_model_') and f.endswith('.pt')]
        
        if len(model_files) <= keep:
            return

        # Extrai scores dos nomes dos arquivos
        models_with_scores = []
        for f in model_files:
            try:
                score = float(f.split('_score_')[1].split('_')[0])
                models_with_scores.append((f, score))
            except:
                continue

        # Ordena do melhor para o pior
        models_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Remove os mais antigos além do limite
        for f, _ in models_with_scores[keep:]:
            os.remove(os.path.join(self.best_model_dir, f))
            
    def cleanup_unused_models(self, current_agents: List[int], model_prefix: str = "evolutionary_trash_collector_agent"):
        """Remove modelos de agentes não utilizados"""
        try:
            # Lista todos os arquivos de modelo
            model_files = [f for f in os.listdir(self.save_dir) 
                        if f.startswith(model_prefix) and f.endswith('.pt')]
            
            # Identifica os agentes atuais (arquivos que devem ser mantidos)
            current_files = [f"{model_prefix}_{i}_" for i in current_agents]
            
            for model_file in model_files:
                # Verifica se o arquivo não pertence a um agente atual
                if not any(prefix in model_file for prefix in current_files):
                    try:
                        os.remove(os.path.join(self.save_dir, model_file))
                        # Remove também o arquivo de metadados correspondente
                        metadata_file = model_file.replace('.pt', '.json')
                        if os.path.exists(os.path.join(self.save_dir, metadata_file)):
                            os.remove(os.path.join(self.save_dir, metadata_file))
                        logger.info(f"Removido modelo não utilizado: {model_file}")
                    except Exception as e:
                        logger.error(f"Erro ao remover {model_file}: {e}")
                        
        except Exception as e:
            logger.error(f"Erro na limpeza de modelos: {e}")
# ==================== HELPER FUNCTIONS ====================
def normalize_position(pos: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
    """Normaliza coordenadas para o intervalo [0,1]
    
    Args:
        pos: Tupla com coordenadas (x, y)
        width: Largura do ambiente
        height: Altura do ambiente
        
    Returns:
        Tupla com coordenadas normalizadas
    """
    return (pos[0] / width, pos[1] / height)

def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calcula distância euclidiana entre dois pontos
    
    Args:
        pos1: Primeiro ponto (x, y)
        pos2: Segundo ponto (x, y)
        
    Returns:
        Distância entre os pontos
    """
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def downsample_data(data: List[float], max_points: int = 500) -> List[float]:
    """Reduz o número de pontos em um conjunto de dados para plotagem eficiente
    
    Args:
        data: Lista de valores a serem reduzidos
        max_points: Número máximo de pontos desejados
        
    Returns:
        Lista reduzida de valores
    """
    if len(data) > max_points:
        factor = len(data) // max_points
        return [np.mean(data[i:i+factor]) for i in range(0, len(data), factor)]
    return data

class CheckpointManager:
    """Gerencia pontos de recuperação do treinamento"""
    
    def __init__(self, checkpoint_dir="checkpoints", max_saves=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_saves = max_saves
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._setup_safe_loading()

    def _setup_safe_loading(self):
        """Configura globais seguros para torch.load"""
        try:
            from numpy.core.multiarray import scalar
            import torch.serialization
            torch.serialization.add_safe_globals([scalar])
        except ImportError as e:
            logger.warning(f"Não foi possível configurar carregamento seguro: {e}")

    def save_checkpoint(self, trainer, episode=None):
        """Salva estado completo do treinamento"""
        checkpoint = {
            'model_state': trainer.agent.state_dict(),
            'optimizer_state': trainer.optimizer.state_dict(),
            'epsilon': trainer.epsilon,
            'episode': episode if episode else trainer.steps_done,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'scores': getattr(trainer, 'current_scores', []),
            'metadata': {
                'framework': 'pytorch',
                'training_phase': 'intermediate'
            }
        }

        filename = f"checkpoint_ep{checkpoint['episode']}_{checkpoint['timestamp']}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        torch.save(checkpoint, path)
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint salvo em {path}")
        return path

    def load_latest_checkpoint(self, trainer):
        """Carrega o checkpoint mais recente"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                    if f.startswith('checkpoint_') and f.endswith('.pt')]
        
        if not checkpoints:
            return False

        # Encontra o checkpoint mais recente
        latest = max(checkpoints, key=lambda x: os.path.getmtime(
            os.path.join(self.checkpoint_dir, x)))
        
        path = os.path.join(self.checkpoint_dir, latest)
        
        try:
            checkpoint = torch.load(path, weights_only=True, map_location='cpu')
            
            trainer.agent.load_state_dict(checkpoint['model_state'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            trainer.epsilon = checkpoint['epsilon']
            trainer.steps_done = checkpoint.get('episode', trainer.steps_done)
            
            if hasattr(trainer, 'current_scores'):
                trainer.current_scores = checkpoint.get('scores', [])
            
            logger.info(f"Checkpoint carregado: {latest}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar checkpoint: {e}")
            return False

    def _cleanup_old_checkpoints(self):
        """Remove checkpoints antigos"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                    if f.startswith('checkpoint_') and f.endswith('.pt')]
        
        if len(checkpoints) <= self.max_saves:
            return

        # Ordena por data de modificação (mais antigo primeiro)
        checkpoints.sort(key=lambda x: os.path.getmtime(
            os.path.join(self.checkpoint_dir, x)))
        
        # Remove os mais antigos
        for old_checkpoint in checkpoints[:-self.max_saves]:
            os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))

# ==================== GAME CLASSES ====================
class TrashCollectionGame:
    """Ambiente de jogo para coleta de lixo com aprendizado por reforço
    
    Atributos:
        width: Largura do ambiente
        height: Altura do ambiente
        agent_pos: Posição atual do agente
        charger_pos: Posição do carregador
        trash_positions: Lista de posições de lixo
        battery: Nível atual da bateria
        collected_trash: Quantidade de lixo coletado
        visited_map: Mapa de áreas visitadas
        memory_vector: Vetor de memória para posições de lixo
    """
    
    def __init__(self, agent_color: Tuple[int, int, int] = (70, 130, 180), render: bool = True):
        """Inicializa o ambiente de jogo
        
        Args:
            agent_color: Cor RGB do agente
            render: Flag para habilitar renderização
        """
        self.frozen = False
        self._initialize_game_parameters(agent_color)
        self.render_flag = render
        if self.render_flag:
            self._setup_pygame()
        self.reset()
        self.memory_system = EnhancedMemory()
    def _initialize_game_parameters(self, agent_color: Tuple[int, int, int]):
        """Configura parâmetros iniciais do jogo"""
        self.width = Config.GAME_WIDTH
        self.height = Config.GAME_HEIGHT
        self.agent_pos = [self.width // 2, self.height // 2]
        self.charger_pos = [self.width - 50, self.height - 50]
        self.agent_color = agent_color
        self.actions = {
            0: [0, -5],  # Up
            1: [0, 5],   # Down
            2: [-5, 0],  # Left
            3: [5, 0],   # Right
            4: [0, 0]    # Stay
        }
    def freeze_agent(self):
        """Congela o agente, impedindo qualquer movimento"""
        self.frozen = True
        self.actions = {  # Sobrescreve as ações para movimento zero
            0: [0, 0],   # Up 
            1: [0, 0],    # Down
            2: [0, 0],    # Left
            3: [0, 0],    # Right
            4: [0, 0]     # Stay
        }
        # Força posição atual (elimina qualquer movimento residual)
        self.agent_pos = [int(self.agent_pos[0]), int(self.agent_pos[1])]
    def _setup_pygame(self):
        """Configura elementos do Pygame"""
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 24)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Trash Collection Game")
        
    def _generate_trash(self, n: int) -> List[Tuple[int, int]]:
        """Gera lixos em posições aleatórias sem sobreposição
        
        Args:
            n: Número de lixos a serem gerados
            
        Returns:
            Lista de posições de lixo
        """
        trash_positions = []
        attempts = 0
        max_attempts = n * 10  # Limite para evitar loops infinitos
        
        while len(trash_positions) < n and attempts < max_attempts:
            attempts += 1
            x = random.randint(20, self.width - 20)
            y = random.randint(20, self.height - 20)
            new_trash = (x, y)

            if self._is_valid_trash_position(new_trash, trash_positions):
                trash_positions.append(new_trash)
                
        if len(trash_positions) < n:
            logger.warning(f"Apenas {len(trash_positions)}/{n} lixos gerados devido a restrições de posição")
            
        return trash_positions
    
    def update_memory(self, agent_idx: int):
        """Atualiza a memória apenas com o carregador"""
        # Remove qualquer verificação de lixo
        charger_dist = calculate_distance(self.agent_positions[agent_idx], self.charger_pos)
        charger_importance = 1 - (charger_dist / (self.width * 0.5))
        self.memory_systems[agent_idx].add(self.charger_pos, "charger", charger_importance)
        
    def _is_valid_trash_position(self, new_trash: Tuple[int, int], 
                            existing_trash: List[Tuple[int, int]]) -> bool:
        """Verifica se a posição do lixo é válida
        
        Args:
            new_trash: Posição do novo lixo
            existing_trash: Lista de lixos existentes
            
        Returns:
            True se a posição é válida, False caso contrário
        """
        # Verifica distância para outros lixos
        for trash in existing_trash:
            if calculate_distance(new_trash, trash) <= Config.MIN_TRASH_DISTANCE:
                return False
                
        # Verifica distância para o carregador
        charger_dist = calculate_distance(new_trash, self.charger_pos)
        return charger_dist > Config.MIN_TRASH_DISTANCE

    def reset(self) -> np.ndarray:
        """Reseta o jogo para o estado inicial
        
        Returns:
            Estado inicial do jogo
        """
        self.current_scores = [0] * self.num_agents
        self.frozen_states = {i: False for i in range(self.num_agents)} 
        self.agent_pos = [self.width // 2, self.height // 2]
        self.trash_positions = self._generate_trash(0)
        self.battery = Config.MAX_BATTERY
        self.collected_trash = 0
        self.visited_map = np.zeros((self.width // 10, self.height // 10))
        self.memory_vector = np.zeros(Config.MEMORY_SIZE * 3)
        return self.get_state()

    def get_visible_objects(self) -> Tuple[List[Tuple[int, int]], bool, Dict[int, List[Tuple[float, Tuple[int, int]]]]]:
        """Retorna TODOS os objetos do mapa (sem restrição de FOV)"""
        sector_trash = {i: [] for i in range(len(Config.SECTOR_ANGLES) - 1)}

        # Processa TODOS os lixos (sem verificação de distância)
        for trash in self.trash_positions:
            angle = math.degrees(math.atan2(trash[1]-self.agent_pos[1], 
                                        trash[0]-self.agent_pos[0])) % 360
            sector = self._get_sector(angle)
            dist = calculate_distance(self.agent_pos, trash)
            sector_trash[sector].append((dist, trash))

        # Ordena por distância em cada setor
        for sector in sector_trash:
            sector_trash[sector].sort()

        # Carregador sempre visível
        charger_visible = True

        return self.trash_positions.copy(), charger_visible, sector_trash

    def _get_sector(self, angle: float) -> int:
        """Determina o setor angular para um ângulo dado
        
        Args:
            angle: Ângulo em graus
            
        Returns:
            Índice do setor angular
        """
        for i in range(len(Config.SECTOR_ANGLES) - 1):
            if Config.SECTOR_ANGLES[i] <= angle < Config.SECTOR_ANGLES[i + 1]:
                return i
        return 0

    def get_state(self, agent_idx: int) -> np.ndarray:
        """Retorna o estado atual do agente com melhor representação visual"""
        agent_pos = self.agent_positions[agent_idx]
        
        # Informações básicas do agente
        state = [
            agent_pos[0] / self.width,  # Posição X normalizada
            agent_pos[1] / self.height,  # Posição Y normalizada
            self.batteries[agent_idx] / Config.MAX_BATTERY  # Bateria normalizada
        ]
        
        # Informação sobre o carregador (sempre visível)
        charger_dist = calculate_distance(agent_pos, self.charger_pos)
        charger_dir = math.atan2(self.charger_pos[1]-agent_pos[1], 
                            self.charger_pos[0]-agent_pos[0])
        state.extend([
            charger_dist / math.sqrt(self.width**2 + self.height**2),  # Distância normalizada
            math.sin(charger_dir),  # Direção X
            math.cos(charger_dir)   # Direção Y
        ])
        
        # Setores para detecção de lixo (8 setores)
        sector_info = []
        for angle in range(0, 360, 45):  # 8 setores de 45 graus
            # Encontra o lixo mais próximo neste setor
            min_dist = float('inf')
            for trash in self.trash_positions:
                trash_angle = math.degrees(math.atan2(trash[1]-agent_pos[1],trash[0]-agent_pos[0])) % 360
                if angle <= trash_angle < angle + 45:
                    dist = calculate_distance(agent_pos, trash)
                    if dist < min_dist:
                        min_dist = dist
            
            # Adiciona informações do setor
            if min_dist != float('inf'):
                sector_info.extend([
                    1,  # Há lixo neste setor
                    1 - (min_dist / math.sqrt(self.width**2 + self.height**2))  # Proximidade
                ])
            else:
                sector_info.extend([0, 0])  # Sem lixo no setor
        
        # Nova memória - recupera memórias relevantes
        relevant_memories = self.memory_system.recall(self.agent_pos, self.width * 0.7)
        
        # Codifica as memórias mais importantes (até Config.MEMORY_SIZE)
        memory_encoding = []
        relevant_memories = self.memory_systems[agent_idx].recall(
            current_pos=self.agent_positions[agent_idx],
            radius=self.width * 0.7
        )
        
        # Preenche com informações do carregador (se existir)
        for mem in relevant_memories[:Config.MEMORY_SIZE]:  # No máximo MEMORY_SIZE memórias
            memory_encoding.extend([
                mem['position'][0] / self.width,      # Pos X normalizada
                mem['position'][1] / self.height,     # Pos Y normalizada
                0,                                    # Tipo (0=carregador)
                mem['importance']                     # Importância
            ])
        
        # Preenche o resto com zeros
        while len(memory_encoding) < Config.MEMORY_SIZE * 4:
            memory_encoding.extend([0, 0, 0, 0])
        
        state.extend(memory_encoding[:Config.MEMORY_SIZE * 4])
        
        return np.array(state, dtype=np.float32)

    def _add_sector_info_to_state(self, state: List[float], 
                                sector_trash: Dict[int, List[Tuple[float, Tuple[int, int]]]]):
        """Adiciona informações dos setores ao estado
        
        Args:
            state: Lista representando o estado atual
            sector_trash: Dicionário de lixos por setor
        """
        for sector in range(8):
            if sector_trash[sector]:
                closest_dist, _ = sector_trash[sector][0]
                state.extend([
                    1.0 - (closest_dist),  # Proximidade normalizada
                    1  # Flag de presença
                ])
            else:
                state.extend([0.0, 0])

    def update_memory(self):
        """Atualiza a memória de posições de lixo"""
        decay_factor = 0.9
        self.memory_vector[2::3] *= decay_factor

        visible_trash, _, _ = self.get_visible_objects()
        for trash in visible_trash:
            self._update_memory_with_trash(trash)

    def _update_memory_with_trash(self, trash: Tuple[int, int]):
        """Atualiza a memória para uma posição de lixo específica
        
        Args:
            trash: Posição do lixo a ser memorizada
        """
        found = False
        trash_norm = normalize_position(trash, self.width, self.height)
        
        # Procura por lixo já memorizado
        for i in range(0, len(self.memory_vector), 3):
            mem_x, mem_y = self.memory_vector[i], self.memory_vector[i+1]
            if (abs(mem_x - trash_norm[0]) < 0.05 and 
                abs(mem_y - trash_norm[1]) < 0.05):
                self.memory_vector[i:i+3] = [*trash_norm, 1.0]
                found = True
                break

        # Adiciona novo lixo se não encontrado
        if not found:
            weakest_pos = np.argmin(self.memory_vector[2::3]) * 3
            self.memory_vector[weakest_pos:weakest_pos+3] = [*trash_norm, 1.0]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Executa um passo no ambiente
        
        Args:
            action: Ação a ser executada (0-4)
            
        Returns:
            Tupla contendo:
            - Novo estado
            - Recompensa
            - Flag se episódio terminou
        """
        # 1. Primeiro verifica bateria
        if self.battery <= 0 and not self.frozen:
            self.freeze_agent()
            return self.get_state(), Config.BATTERY_PENALTY, True
        
        # 2. Depois processa movimento (se não estiver frozen)
        if not self.frozen:
            self._move_agent(action)
            self._update_battery(action)
        
        # 3. Calcula recompensas
        reward, done = self._calculate_rewards_and_done()
        return self.get_state(), reward, done
    def _move_agent(self, action: int):
        """Move o agente baseado na ação
        
        Args:
            action: Ação a ser executada (0-4)
        """
        if self.frozen:
            return
        
        move = self.actions[action]
        self.agent_pos[0] = np.clip(self.agent_pos[0] + move[0], 0, self.width)
        self.agent_pos[1] = np.clip(self.agent_pos[1] + move[1], 0, self.height)

    def _update_battery(self, action: int):
        """Atualiza o nível da bateria
        
        Args:
            action: Ação executada (afeta o consumo de bateria)
        """
        drain = (Config.BATTERY_DRAIN_MOVE if action != 4 
                else Config.BATTERY_DRAIN_IDLE)
        self.battery = max(0, self.battery - drain)

    def _calculate_rewards_and_done(self, agent_idx: Optional[int] = None) -> Tuple[float, bool]:
        """Calcula recompensas e verifica condições de término"""
        reward = 0.0
        done = False
        
        # Determina valores com base no modo (single/multi-agent)
        if agent_idx is not None:
            agent_pos = self.agent_positions[agent_idx]
            battery_level = self.batteries[agent_idx]
        else:
            agent_pos = self.agent_pos
            battery_level = self.battery
        
        battery_urgency = 1 - (battery_level / Config.MAX_BATTERY)
        
        # Recompensa por coletar lixo
        reward += self._calculate_trash_reward(agent_idx, agent_pos, battery_urgency)
        
        # Recompensa por recarregar
        reward += self._calculate_charging_reward(agent_idx, agent_pos, battery_urgency)
        
        # Penalidade por bateria zerada
        if battery_level <= 0:
            reward += Config.BATTERY_PENALTY
            done = True
            
        # Recompensa por completar a coleta
        if not self.trash_positions:
            reward += Config.COMPLETION_BONUS
            done = True
            
        # Penalidade por movimento
        reward += Config.MOVEMENT_PENALTY
        
        
        return reward, done

    def _calculate_trash_reward(self, agent_idx: int, agent_pos: List[int], battery_urgency: float) -> float:
        """Calcula recompensas por interação com lixo (aproximação/coleta/afastamento)
        
        Args:
            agent_idx: Índice do agente
            agent_pos: Posição atual do agente [x, y]
            battery_urgency: Urgência de bateria (0-1, onde 1 é bateria crítica)
            
        Returns:
            Recompensa total por interações com lixo
        """
        reward = 0
        min_dist = float('inf')
        nearest_trash = None
        battery_level = 1 - battery_urgency  # Converter para nível de bateria (0-1)
        
        # 1. Verifica coleta de lixo
        reward += self._calculate_collection_reward(agent_idx, agent_pos, battery_level)
        
        # 2. Calcula recompensa por aproximação e encontra lixo mais próximo
        reward, min_dist, nearest_trash = self._calculate_approach_reward(
            agent_idx, agent_pos, battery_level, reward, min_dist, nearest_trash)
        
        # 3. Aplica penalidades por afastamento
        reward = self._calculate_avoidance_penalty(
            agent_idx, agent_pos, battery_level, nearest_trash, min_dist, reward)
        
        # Atualiza última posição para cálculo no próximo frame
        self.last_positions[agent_idx] = agent_pos.copy()
        
        return reward

    def _calculate_collection_reward(self, agent_idx: int, agent_pos: List[int], battery_level: float) -> float:
        """Calcula recompensa por coletar lixo"""
        reward = 0
        for trash in self.trash_positions[:]:
            dist = calculate_distance(agent_pos, trash)
            if dist < Config.TRASH_COLLECTION_RADIUS:
                self.trash_positions.remove(trash)
                base_reward = Config.TRASH_COLLECTED_REWARD
                
                # Bônus se estiver com bateria boa (>70%)
                if battery_level > 0.7:
                    base_reward *= Config.BATTERY_BONUS_MULTIPLIER
                    
                reward += base_reward
                logger.info(f"Agente {agent_idx} coletou lixo! +{base_reward:.1f} pts")
                break
        return reward

    def _calculate_approach_reward(self, agent_idx: int, agent_pos: List[int], 
                                battery_level: float, current_reward: float,
                                current_min_dist: float, current_nearest_trash: Optional[Tuple[int, int]]):
        """Calcula recompensa por aproximação do lixo"""
        reward = current_reward
        min_dist = current_min_dist
        nearest_trash = current_nearest_trash
        
        for trash in self.trash_positions:
            dist = calculate_distance(agent_pos, trash)
            
            if dist < Config.TRASH_APPROACH_REWARD_RANGE * Config.TRASH_COLLECTION_RADIUS:
                # Recompensa base por aproximação
                proximity_reward = Config.APPROACH_TRASH_REWARD * (
                    1 - (dist / (Config.TRASH_APPROACH_REWARD_RANGE * Config.TRASH_COLLECTION_RADIUS)))
                
                # Bônus adicional por bateria cheia
                battery_bonus = Config.BATTERY_BONUS * battery_level
                
                total_reward = proximity_reward + battery_bonus
                
                # Penaliza se bateria estiver crítica (<10%)
                if battery_level < 0.1:
                    total_reward *= 0.3
                    
                reward += total_reward
                
                # Atualiza lixo mais próximo
                if dist < min_dist:
                    min_dist = dist
                    nearest_trash = trash
                    
        return reward, min_dist, nearest_trash

    def _calculate_avoidance_penalty(self, agent_idx: int, agent_pos: List[int],
                                battery_level: float, nearest_trash: Optional[Tuple[int, int]],
                                min_dist: float, current_reward: float) -> float:
        """Calcula penalidades por afastamento do lixo"""
        reward = current_reward
        
        if nearest_trash and min_dist < Config.TRASH_APPROACH_REWARD_RANGE * Config.TRASH_COLLECTION_RADIUS:
            last_dist = calculate_distance(self.last_positions[agent_idx], nearest_trash)
            
            # Se afastou do lixo
            if min_dist > last_dist:
                # Penalidade base por afastamento
                distance_penalty = Config.AWAY_FROM_TRASH_PENALTY * (
                    (min_dist - last_dist) / Config.TRASH_COLLECTION_RADIUS)
                
                # Penalidade adicional se bateria estiver alta (>50%)
                if battery_level > 0.5:
                    distance_penalty *= Config.HIGH_BATTERY_AVOIDANCE_MULTIPLIER
                    
                reward += distance_penalty
                logger.debug(f"Agente {agent_idx} afastou-se do lixo: {distance_penalty:.2f} pts")
                
        return reward

    def _calculate_charging_reward(self) -> float:
        """Calcula recompensa por recarregar
        
        Returns:
            Recompensa total por recarga
        """
        charger_dist = calculate_distance(self.agent_pos, self.charger_pos)
        if charger_dist < Config.CHARGER_RADIUS:
            charge_amount = min(2.0, Config.MAX_BATTERY - self.battery)
            self.battery = min(Config.MAX_BATTERY, self.battery + charge_amount)
            return Config.CHARGER_REWARD
        return 0.0



    def render(self, episode: Optional[int] = None, reward: Optional[float] = None):
        """Renderiza o estado atual do jogo
        
        Args:
            episode: Número do episódio atual
            reward: Recompensa acumulada
        """
        if not self.render_flag:
            return
        
        try:
            
            # Tratamento de eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            self.screen.fill((240, 240, 240))
            
            # Desenha carregador
            pygame.draw.rect(self.screen, (0, 200, 0), 
                            (self.charger_pos[0]-15, self.charger_pos[1]-15, 30, 30))
            
            # Desenha TODOS os lixos (agora todos são visíveis)
            for trash in self.trash_positions:
                pygame.draw.circle(self.screen, (255, 0, 0), trash, 8)
            
            # Desenha agentes (sem FOV)
            for idx, agent_pos in enumerate(self.agent_positions):
                pygame.draw.circle(self.screen, self.agent_colors[idx], agent_pos, 12)  
            
        except pygame.error as e:
            logger.error(f"Erro de renderização: {e}")
            self.render_flag = False

    def _draw_charger(self):
        """Desenha a estação de carregamento"""
        charger_rect = pygame.Rect(self.charger_pos[0]-15, self.charger_pos[1]-15, 30, 30)
        pygame.draw.rect(self.screen, (0, 200, 0), charger_rect)
        pygame.draw.line(self.screen, (0, 0, 0), (charger_rect.centerx-5, charger_rect.centery),(charger_rect.centerx+5, charger_rect.centery), 3)
        pygame.draw.line(self.screen, (0, 0, 0), (charger_rect.centerx, charger_rect.centery-5),(charger_rect.centerx, charger_rect.centery+5), 3)

    def _draw_battery_status(self):
        """Desenha o status da bateria"""
        battery_color = (
            int(max(0, min(255, 255 * (2 - self.battery / 50)))),
            int(max(0, min(255, 255 * (self.battery / 50)))),
            0
        )
        battery_text = self.font.render(f'Bateria: {int(self.battery)}%', True, battery_color)
        self.screen.blit(battery_text, (10, 10))

    def _draw_trash(self):
        """Desenha os lixos no ambiente"""
        visible_trash, charger_visible, _ = self.get_visible_objects()
        
        # Desenha lixos visíveis
        for trash in visible_trash:
            pygame.draw.circle(self.screen, (255, 0, 0), trash, 8)
            
        # Desenha lixos não visíveis
        for trash in self.trash_positions:
            if trash not in visible_trash:
                pygame.draw.circle(self.screen, (139, 69, 19), trash, 8)
                
        # Destaca carregador se visível
        if charger_visible:
            pygame.draw.rect(self.screen, (255, 0, 0), (*self.charger_pos, 30, 30), 2)

    def _draw_agent(self):
        """Desenha o agente"""
        pygame.draw.circle(self.screen, self.agent_color, self.agent_pos, 12)
        
        # Destaca se estiver carregando
        charger_dist = calculate_distance(self.agent_pos, self.charger_pos)
        if charger_dist < Config.CHARGER_RADIUS:
            pygame.draw.circle(self.screen, (255, 255, 0), self.agent_pos, 15, 2)

    def _draw_info_text(self, episode: Optional[int], reward: Optional[float]):
        """Desenha informações textuais na tela"""
        trash_text = self.font.render(f'Lixo: {self.collected_trash}/10', True, (0, 0, 0))
        self.screen.blit(trash_text, (10, 40))
        
        if episode is not None and reward is not None:
            info_text = self.font.render(
                f'Episódio: {episode} | Recompensa: {reward:.1f}', True, (0, 0, 0))
            self.screen.blit(info_text, (self.width // 2 - 100, 10))

# ==================== NEURAL NETWORK CLASSES ====================
class CuriosityModule(nn.Module):
    """Módulo de curiosidade para aprendizado intrínseco"""
    
    def __init__(self, input_size: int, hidden_size: int = 32):
        """Inicializa o módulo de curiosidade
        
        Args:
            input_size: Tamanho do estado de entrada
            hidden_size: Tamanho da camada oculta
        """
        super().__init__()
        self.forward_model = nn.Sequential(
            nn.Linear(input_size + 5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, 
            next_state: torch.Tensor) -> torch.Tensor:
        """Calcula recompensa de curiosidade
        
        Args:
            state: Estado atual
            action: Ação tomada
            next_state: Próximo estado
            
        Returns:
            Recompensa de curiosidade
        """
        action_onehot = torch.zeros(action.size(0), 5, device=action.device)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        concatenated = torch.cat([state, action_onehot], dim=1)
        predicted_next_state = self.forward_model(concatenated)
        return F.mse_loss(predicted_next_state, next_state, reduction='none').mean(1)

class EnhancedSNN(nn.Module):
    def __init__(self, with_curiosity=None):
        super().__init__()
        
        # 1. Normalização de entrada
        self.input_norm = nn.LayerNorm(Config.STATE_SIZE)
        
        # 2. Processamento visual (STATE_SIZE → 512)
        self.visual_processing = nn.Sequential(
            nn.Linear(Config.STATE_SIZE, 512),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(512),
            nn.Dropout(0.1))
        
        # 3. Mecanismo de atenção
        self.attention = nn.MultiheadAttention(
            embed_dim=512,  # Deve corresponder à saída do visual_processing
            num_heads=8,
            dropout=0.1,
            batch_first=True)
        
        # 4. Blocos residuais com dimensões corrigidas
        self.res_block1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(256),
            nn.Dropout(0.2))
            
        self.res_block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
            nn.Dropout(0.2))
            
        self.res_block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(64),
            nn.Dropout(0.1))
        
        # 5. Camadas SNN (corrigidas para usar fc_snn em vez de fc)
        self.fc_snn0 = nn.Linear(64, 64)
        self.lif0 = snn.Leaky(beta=0.95, threshold=0.8, reset_mechanism="zero")
        
        self.fc_snn1 = nn.Linear(64, 64)
        self.lif1 = snn.Leaky(beta=0.92, threshold=0.85, reset_mechanism="zero")
        
        self.fc_snn2 = nn.Linear(64, 32)
        self.lif2 = snn.Leaky(beta=0.9, threshold=0.9, reset_mechanism="zero")
        
        self.fc_snn3 = nn.Linear(32, 32)
        self.lif3 = snn.Leaky(beta=0.85, threshold=0.95, reset_mechanism="zero")
        
        # 6. Camada de saída
        self.fc_out = nn.Linear(32, 5)
        
    def forward(self, x: torch.Tensor, time_window: int = 10) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Normalização de entrada
        x = self.input_norm(x)
        
        # Processamento visual
        visual_out = self.visual_processing(x)
        
        # Atenção
        attn_out, _ = self.attention(
            visual_out.unsqueeze(1,), # Adiciona dimensão de batch 
            visual_out.unsqueeze(1),
            visual_out.unsqueeze(1))
        attn_out = attn_out.squeeze(1) 
        
        # Blocos residuais (corrigido o skip connection)
        res1 = self.res_block1(attn_out)
        res1 = res1 + attn_out[:, :256]  # Skip connection ajustada
        
        res2 = self.res_block2(res1)
        res2 = res2 + res1[:, :128]  # Skip connection ajustada
        
        res3 = self.res_block3(res2)
        res3 = res3 + res2[:, :64]  # Skip connection ajustada
        
        # Processamento SNN
        mem0 = self.lif0.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spikes = torch.zeros(x.size(0), 5, device=x.device)
        
        for _ in range(time_window):
            x0 = self.fc_snn0(res3)
            spk0, mem0 = self.lif0(x0, mem0)
            
            x1 = self.fc_snn1(spk0)
            spk1, mem1 = self.lif1(x1, mem1)
            
            x2 = self.fc_snn2(spk1)
            spk2, mem2 = self.lif2(x2, mem2)
            
            x3 = self.fc_snn3(spk2)
            spk3, mem3 = self.lif3(x3, mem3)
            
            spikes += torch.sigmoid(self.fc_out(spk3))
            
        return spikes / time_window

# ==================== REPLAY BUFFER CLASSES ====================
class Transition(namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))):
    """Tupla nomeada para armazenar transições de experiência"""
    pass

class PrioritizedReplayBuffer:
    """Buffer de replay com priorização para aprendizado por reforço"""
    
    def __init__(self, capacity: int = 50000, alpha: float = 0.6, 
                beta_start: float = 0.4, beta_frames: int = 100000):
        """Inicializa o buffer
        
        Args:
            capacity: Capacidade máxima do buffer
            alpha: Parâmetro de priorização (0 = uniforme)
            beta_start: Valor inicial para compensação de bias
            beta_frames: Número de frames para anelar beta
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 16
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def __len__(self) -> int:
        """Retorna o tamanho atual do buffer"""
        return len(self.buffer)

    def add(self, state: np.ndarray, action: int, reward: float, 
        next_state: np.ndarray, done: bool):
        """Adiciona uma transição ao buffer
        
        Args:
            state: Estado atual
            action: Ação tomada
            reward: Recompensa recebida
            next_state: Próximo estado
            done: Flag de término
        """
        state = np.array(state, dtype=np.float32).flatten()
        reward = float(reward)
        next_state = np.array(next_state, dtype=np.float32).flatten()
        done = bool(done)

        max_priority = max(self.priorities) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = Transition(state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        """Amostra um lote de transições
        
        Args:
            batch_size: Tamanho do lote
            
        Returns:
            Tupla contendo:
            - Estados
            - Ações
            - Recompensas
            - Próximos estados
            - Flags de término
            - Índices
            - Pesos
        """
        if len(self.buffer) < batch_size:
            return [], [], [], [], [], [], []

        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        states = np.array([t.state for t in samples])
        actions = np.array([t.action for t in samples])
        rewards = np.array([t.reward for t in samples])
        next_states = np.array([t.next_state for t in samples])
        dones = np.array([t.done for t in samples])
        
        weights = (len(self.buffer) * probs[indices]) ** (-0.4)
        weights /= weights.max()

        return (states, actions, rewards, next_states, dones, 
                indices, np.array(weights, dtype=np.float32))

    def update_priorities(self, indices: List[int], errors: np.ndarray):
        """Atualiza prioridades das transições
        
        Args:
            indices: Índices das transições
            errors: Erros de TD das transições
        """
        for idx, error in zip(indices, errors):
            priority = (abs(error.item()) + 1e-5) ** self.alpha
            # Aumentar prioridade para transições com alta recompensa
            if error.item() > 10.0:
                priority *= 2.0
            self.priorities[idx] = float(priority)

# ==================== TRAINING CLASSES ====================
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
        """Salva o modelo atual"""
        metadata = {
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'save_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'performance': max(self.current_scores) if hasattr(self, 'current_scores') else 0
        }
        
        model_path = self.save_manager.save_model(self.agent, self.model_name, metadata)
        logger.info(f"Modelo salvo em: {model_path}")
        return model_path
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

# ==================== EVOLUTIONARY TRAINER ====================
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
        
        if not hasattr(self, 'generation_best_scores'):
            return [(i, 0) for i in range(self.population_size)]
        
        # Cria lista de (índice, melhor_score) para cada agente
        ranked_agents = [(i, self.generation_best_scores[i]) 
                        for i in range(self.population_size)]
        
        # Ordena do melhor para o pior
        ranked_agents.sort(key=lambda x: x[1], reverse=True)
        
        if not hasattr(self.game, 'current_scores'):
            logger.error("current_scores não existe no game!")
            return [(i, 0) for i in range(self.population_size)]

        if len(self.game.current_scores) != self.population_size:
            logger.error(f"Tamanho incorreto de current_scores: {len(self.game.current_scores)} != {self.population_size}")
            return [(i, 0) for i in range(self.population_size)]
        
        try:
            
            # Usa os scores máximos por episódio de cada agente
            ranked = sorted(
                [(i, float(max(self.generation_scores[i])))  # Pega o melhor score individual do agente
                for i in range(self.population_size)],
                key=lambda x: x[1], 
                reverse=True
            )
            return ranked
        
        
        except Exception as e:
            logger.error(f"Erro ao avaliar agentes: {str(e)}")
            return ranked_agents,[(i, 0) for i in range(self.population_size)]

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

# ==================== COMPETITIVE GAME ====================
class CompetitiveTrashGame(TrashCollectionGame):
    """Ambiente competitivo para múltiplos agentes com coleta de lixo"""

    def __init__(self, num_agents: int = 2, render: bool = True, 
                time_limit: int = Config.TIME_LIMIT):
        """Inicializa o ambiente competitivo
        
        Args:
            num_agents: Número de agentes (deve ser pelo menos 2)
            render: Se True, habilita a renderização gráfica
            time_limit: Tempo limite por episódio em segundos
        """
        
        # 1. Validação de parâmetros
        if num_agents < 2:
            raise ValueError("CompetitiveTrashGame requer pelo menos 2 agentes")
        
        # 2. Configuração básica
        self._num_agents = num_agents  # Nome protegido
        self.trash_count = 3  # Número inicial de lixos
        self.time_limit = time_limit
        self.frozen = False
        self.frozen_states = {i: False for i in range(num_agents)}
        
        # 3. Inicialização da classe pai
        super().__init__(agent_color=(0, 0, 0), render=render)
        logger.info(f"Inicializando CompetitiveTrashGame com {num_agents} agentes")

        # 4. Inicialização de estruturas de agentes
        self.agent_positions = self._initialize_agent_positions()
        self.last_positions = [pos.copy() for pos in self.agent_positions]
        self.batteries = [Config.MAX_BATTERY] * self.num_agents
        
        # 5. Sistemas de memória e aprendizado
        self.memory_systems = [EnhancedMemory() for _ in range(self.num_agents)]
        
        # 6. Configuração visual dos agentes
        self.agent_colors = [
            (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            for _ in range(self.num_agents)
        ]
        
        # 7. Estado do jogo
        self.current_scores = [0] * self.num_agents
        self.best_score = -float('inf')



    @property
    def num_agents(self) -> int:
        """Getter para o número de agentes (propriedade somente leitura)"""
        return self._num_agents

    def _initialize_agent_positions(self) -> List[List[int]]:
        """Inicializa todos os agentes na mesma posição central"""
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Cria uma lista com a mesma posição para todos os agentes
        positions = [[center_x, center_y] for _ in range(self.num_agents)]
        
        logger.debug(f"Posições iniciais: {positions}")
        return positions
    def freeze_agent(self, agent_idx: int):
        """Congela um agente específico temporariamente"""
        self.frozen_states[agent_idx] = True
        # Não precisamos sobrescrever self.actions aqui

    def unfreeze_agent(self, agent_idx: int):
        """Descongela um agente quando condições forem restabelecidas"""
        if self.frozen_states.get(agent_idx, False):
            self.frozen_states[agent_idx] = False
            logger.info(f"Agente {agent_idx} descongelado")
            
    def reset(self) -> List[np.ndarray]:
        
        """Reseta o ambiente para todos os agentes"""
        self.frozen = False
        self.frozen_states = {i: False for i in range(self.num_agents)}  # Reset estados congelados
        logger.debug(f"Resetando jogo para {self.num_agents} agentes")
        self.agent_positions = self._initialize_agent_positions()
        self.trash_positions = self._generate_trash(self.trash_count)
        self.batteries = [Config.MAX_BATTERY] * self.num_agents
        self.current_scores = [0] * self.num_agents
        self.visited_map = np.zeros((self.width // 10, self.height // 10))
        self.last_positions = [pos.copy() for pos in self.agent_positions]
        self.memory_systems = [EnhancedMemory() for _ in range(self.num_agents)]
        logger.info(f"Total de lixos: {len(self.trash_positions)}")
        
        return [self.get_state(i) for i in range(self.num_agents)]
    
    def update_agent_needs(self, agent_idx: int):
        """Atualiza o estado de necessidade do agente"""
        agent_pos = self.agent_positions[agent_idx]
        battery_level = self.batteries[agent_idx] / Config.MAX_BATTERY
        
        # Atualiza memória com objetos visíveis
        visible_trash, _, _ = self.get_visible_objects()
        for trash in visible_trash:
            dist = calculate_distance(agent_pos, trash)
            importance = 1 - (dist / (self.width * 0.5))
            self.memory_systems[agent_idx].add(trash, "trash", importance)
        
        # Adiciona carregador à memória
        charger_dist = calculate_distance(agent_pos, self.charger_pos)
        charger_importance = 1 - (charger_dist / (self.width * 0.5))
        self.memory_systems[agent_idx].add(self.charger_pos, "charger", charger_importance)

    def _get_nearest_trash_distance(self, agent_idx: int) -> float:
        """Retorna a distância para o lixo mais próximo"""
        if not self.trash_positions:
            return self.width  # Valor máximo se não houver lixo
        
        agent_pos = self.agent_positions[agent_idx]
        return min(calculate_distance(agent_pos, trash) for trash in self.trash_positions)

    def get_state(self, agent_idx: int) -> np.ndarray:
        """Obtém o estado para um agente específico no modo competitivo
        
        Args:
            agent_idx: Índice do agente (0 a num_agents-1)
            
        Returns:
            Array numpy representando o estado do agente, incluindo:
            - Posição normalizada
            - Bateria normalizada
            - Informações do carregador
            - Detecção por setores
            - Memórias relevantes
        """
        # 1. Verificação de segurança
        if agent_idx < 0 or agent_idx >= self.num_agents:
            raise ValueError(f"Índice de agente inválido: {agent_idx} (deve estar entre 0 e {self.num_agents-1})")
        
        # Obtém a posição e bateria do agente específico
        agent_pos = self.agent_positions[agent_idx]
        battery_level = self.batteries[agent_idx]
        
        # 2. Informações básicas do agente (normalizadas)
        state = [
            agent_pos[0] / self.width,               # Posição X (0-1)
            agent_pos[1] / self.height,              # Posição Y (0-1)
            battery_level / Config.MAX_BATTERY       # Nível de bateria (0-1)
        ]
        
        # 3. Informações sobre o carregador (sempre visível)
        charger_dist = calculate_distance(agent_pos, self.charger_pos)
        charger_dir = math.atan2(self.charger_pos[1]-agent_pos[1], 
                            self.charger_pos[0]-agent_pos[0])
        state.extend([
            charger_dist / math.sqrt(self.width**2 + self.height**2),  # Distância normalizada
            math.sin(charger_dir),  # Direção X (seno do ângulo)
            math.cos(charger_dir)   # Direção Y (cosseno do ângulo)
        ])
        
        # 4. Detecção por setores (8 setores de 45 graus cada)
        sector_info = []
        for angle in range(0, 360, 45):  # 0, 45, 90, 135, 180, 225, 270, 315
            min_dist = float('inf')
            
            # Encontra o lixo mais próximo neste setor
            for trash in self.trash_positions:
                # Calcula ângulo do lixo em relação ao agente
                trash_angle = math.degrees(math.atan2(trash[1]-agent_pos[1], 
                                            trash[0]-agent_pos[0])) % 360
                
                # Verifica se está no setor atual
                if angle <= trash_angle < angle + 45:
                    dist = calculate_distance(agent_pos, trash)
                    if dist < min_dist:
                        min_dist = dist
            
            # Adiciona informações do setor ao estado
            if min_dist != float('inf'):
                # Presença (1) + Proximidade normalizada (0-1, onde 1 é mais próximo)
                sector_info.extend([1, 1 - (min_dist / math.sqrt(self.width**2 + self.height**2))])
            else:
                # Sem lixo no setor
                sector_info.extend([0, 0])
        
        state.extend(sector_info)
        
        # 5. Memórias relevantes (usando o novo sistema EnhancedMemory)
        relevant_memories = self.memory_systems[agent_idx].recall(
            current_pos=agent_pos,
            radius=self.width * 0.7  # Raio de memória (70% da largura do mapa)
        )
        
        # 6. Codifica as memórias no estado
        memory_encoding = []
        for mem in relevant_memories[:Config.MEMORY_SIZE]:  # Pega no máximo MEMORY_SIZE memórias
            memory_encoding.extend([
                mem['position'][0] / self.width,     # Posição X normalizada
                mem['position'][1] / self.height,    # Posição Y normalizada
                1 if mem['type'] == 'trash' else 0,  # Tipo (1 para lixo, 0 para carregador)
                mem['importance']                     # Importância (0-1)
            ])
        
        # Preenche com zeros se não houver memórias suficientes
        while len(memory_encoding) < Config.MEMORY_SIZE * 4:
            memory_encoding.extend([0, 0, 0, 0])
        
        state.extend(memory_encoding[:Config.MEMORY_SIZE * 4])
        return np.array(state, dtype=np.float32)
    
    
    def check_types(func):
        def wrapper(*args, **kwargs):
            self = args[0]
            if not all(isinstance(pos, (list, np.ndarray)) for pos in self.agent_positions):
                self.agent_positions = [list(pos) for pos in self.agent_positions]
            return func(*args, **kwargs)
        return wrapper

    # Aplique aos métodos críticos
    @check_types
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool]]:
        """Executa um passo para todos os agentes"""
        next_states, rewards, dones = [], [], []
        
        for i, action in enumerate(actions):
            # Verifica se deve descongelar (bateria recarregou)
            if self.frozen_states.get(i, False) and self.batteries[i] > Config.MAX_BATTERY * 0.2:  # 20% de bateria
                self.unfreeze_agent(i)
            
            # Se congelado, ignora ação
            if self.frozen_states.get(i, False):
                next_states.append(self.get_state(i))
                rewards.append(0)
                dones.append(True)
                continue
                
            # Calcula distância para o carregador ANTES de mover o agente
            charger_dist = calculate_distance(self.agent_positions[i], self.charger_pos)
            battery_level = self.batteries[i]
            reward = 0
            
            # Aplica penalidade por ficar no carregador com bateria alta
            if (charger_dist < Config.CHARGER_RADIUS and 
                battery_level > 0.5 * Config.MAX_BATTERY and 
                action == 4):  # Ação 4 é ficar parado
                reward += Config.HIGH_BATTERY_PENALTY * (battery_level / Config.MAX_BATTERY)
            
            
            # Lógica normal para agentes não congelados
            self._move_agent(i, action)
            self._update_battery(i, action)
            
            self.update_agent_needs(i)
            step_reward, done = self._calculate_rewards_and_done(i)
            reward += step_reward
            
            
            next_states.append(self.get_state(i))
            rewards.append(reward)
            dones.append(done)

        return next_states, rewards, dones

    def _move_agent(self, agent_idx: int, action: int):
        """Move um agente específico"""
        move = self.actions[action]
        self.agent_positions[agent_idx][0] = np.clip(
            self.agent_positions[agent_idx][0] + move[0], 0, self.width)
        self.agent_positions[agent_idx][1] = np.clip(
            self.agent_positions[agent_idx][1] + move[1], 0, self.height)

    def _update_battery(self, agent_idx: int, action: int):
        """Atualiza bateria com recarga automática quando no carregador"""
        # Verifica se está no carregador
        charger_dist = calculate_distance(self.agent_positions[agent_idx], self.charger_pos)
        
        if charger_dist < Config.CHARGER_RADIUS:
            # Recarrega mais rápido que o consumo
            charge_amount = Config.BATTERY_DRAIN_MOVE * 1.5  # Recarrega 50% mais rápido que o gasto
            self.batteries[agent_idx] = min(Config.MAX_BATTERY, 
                                        self.batteries[agent_idx] + charge_amount)
        else:
            # Consumo normal de bateria
            drain = (Config.BATTERY_DRAIN_MOVE if action != 4 
                    else Config.BATTERY_DRAIN_IDLE)
            self.batteries[agent_idx] = max(0, self.batteries[agent_idx] - drain)
        
        # Congela se bateria zerar
        if self.batteries[agent_idx] <= 0:
            self.freeze_agent(agent_idx)

    def _calculate_unified_reward(self, agent_idx: int) -> float:
        """Calcula recompensa unificada para um agente com base em:
        - Estado da bateria
        - Proximidade de lixo e carregador
        - Comportamento do agente (movimento/ociosidade)
        
        Args:
            agent_idx: Índice do agente
            
        Returns:
            Recompensa total calculada
        """
        # 1. Obter estado atual do agente
        agent_pos = self.agent_positions[agent_idx]
        battery_level = self.batteries[agent_idx]
        battery_urgency = 1 - (battery_level / Config.MAX_BATTERY)
        reward = 0
        
        # 2. Verificar comportamento do agente
        is_idle = hasattr(self, 'last_actions') and self.last_actions[agent_idx] == 4
        charger_dist = calculate_distance(agent_pos, self.charger_pos)
        nearest_trash_dist = self._get_nearest_trash_distance(agent_idx)
        trash_urgency = 1 - (nearest_trash_dist / (self.width * 0.5)) if self.trash_positions else 0

        # 3. Aplicar penalidades por comportamento inadequado
        # 3.1. Penalidade por ficar parado no carregador com bateria alta
        if is_idle and charger_dist < Config.CHARGER_RADIUS and battery_level > 0.6 * Config.MAX_BATTERY:
            penalty = Config.IDLE_AT_CHARGER_PENALTY * (battery_level / Config.MAX_BATTERY)
            reward += penalty
            logger.debug(f"Agente {agent_idx} - Penalidade por ociosidade no carregador: {penalty:.2f}")

        # 3.2. Penalidade por ficar parado com tarefas urgentes
        urgent_tasks = (battery_urgency > 0.4 or (trash_urgency > 0.7 and battery_urgency < 0.8))
        if is_idle and urgent_tasks:
            penalty = Config.IDLE_WITH_URGENT_TASKS_PENALTY * max(battery_urgency, trash_urgency)
            reward += penalty
            logger.debug(f"Agente {agent_idx} - Penalidade por ociosidade com tarefas urgentes: {penalty:.2f}")

        # 4. Calcular recompensas principais (sem duplicação)
        # 4.1. Recompensa por interação com lixo
        trash_reward = self._calculate_trash_reward(agent_idx, agent_pos, battery_urgency)
        reward += trash_reward
        
        # 4.2. Recompensa por interação com carregador
        charging_reward = self._calculate_charging_reward(agent_idx, agent_pos, battery_urgency)
        reward += charging_reward

        # 5. Aplicar condições especiais
        # 5.1. Penalidade por bateria zerada
        if battery_level <= 0:
            penalty = Config.BATTERY_PENALTY * (1 + battery_urgency)
            reward += penalty
            logger.debug(f"Agente {agent_idx} - Penalidade por bateria zerada: {penalty:.2f}")
        
        # 5.2. Bônus por completar todos os lixos
        if not self.trash_positions:
            reward += Config.COMPLETION_BONUS
            logger.debug(f"Agente {agent_idx} - Bônus por completar missão: {Config.COMPLETION_BONUS:.2f}")

        # 6. Aplicar penalidade por movimento (incentivo à eficiência)
        reward += Config.MOVEMENT_PENALTY

        # 7. Atualizar pontuação do agente e garantir limites
        self.current_scores[agent_idx] += reward

        
        logger.debug(
            f"Agente {agent_idx} - Recompensa total: {reward:.2f} | "
            f"Bateria: {battery_level:.1f}% | "
            f"Distância lixo: {nearest_trash_dist:.1f} | "
            f"Distância carregador: {charger_dist:.1f}"
        )
        
        return reward

    def _calculate_trash_reward(self, agent_idx: int, agent_pos: List[int], battery_urgency: float) -> float:
        """Calcula recompensas por interação com lixo e remove o lixo coletado"""
        reward = 0
        battery_level = 1 - battery_urgency
        
        if not self.trash_positions:
            return reward  # Sem lixo, sem recompensa

        # Encontrar lixo mais próximo e sua distância
        nearest_trash = None
        min_dist = float('inf')
        trash_to_remove = None
        
        for trash in self.trash_positions:
            dist = calculate_distance(agent_pos, trash)
            if dist < min_dist:
                min_dist = dist
                nearest_trash = trash
            if dist < Config.TRASH_COLLECTION_RADIUS:
                trash_to_remove = trash  # Marca lixo para remoção

        # 1. Processar coleta de lixo (prioridade máxima)
        if trash_to_remove is not None:
            base_reward = Config.TRASH_COLLECTED_REWARD
            if battery_level > 0.7:  # Bônus se bateria >70%
                base_reward *= Config.BATTERY_BONUS_MULTIPLIER
            
            # Remove o lixo coletado
            self.trash_positions.remove(trash_to_remove)
            logger.info(f"Agente {agent_idx} coletou lixo na posição {trash_to_remove}! +{base_reward:.1f} pts")
            
            # Atualiza o lixo mais próximo após remoção
            if self.trash_positions:
                nearest_trash, min_dist = self._find_nearest_trash(agent_pos)
            else:
                nearest_trash = None
                min_dist = float('inf')
            
            return base_reward  # Retorna imediatamente após coleta

        # 2. Recompensa/Penalidade por aproximação/afastamento (só se não coletou)
        if hasattr(self, 'last_positions') and nearest_trash is not None:
            last_dist = calculate_distance(self.last_positions[agent_idx], nearest_trash)
            dist_diff = last_dist - min_dist  # Positivo se aproximou, negativo se afastou
            
            if dist_diff > 0:  # Aproximando do lixo
                approach_reward = Config.APPROACH_TRASH_REWARD * dist_diff * self._calculate_urgency_factor(battery_level, min_dist)
                reward += approach_reward
                logger.debug(f"Agente {agent_idx} aproximando do lixo: +{approach_reward:.2f}")
            elif dist_diff < 0:  # Afastando do lixo
                avoidance_penalty = self._calculate_avoidance_penalty(agent_idx, battery_level, min_dist, abs(dist_diff))
                reward += avoidance_penalty
                logger.debug(f"Agente {agent_idx} afastando do lixo: {avoidance_penalty:.2f}")

        # Atualiza última posição
        if not hasattr(self, 'last_positions'):
            self.last_positions = [pos.copy() for pos in self.agent_positions]
        self.last_positions[agent_idx] = agent_pos.copy()
        
        return reward

    def _find_nearest_trash(self, agent_pos: List[int]) -> Tuple[Tuple[int, int], float]:
        """Encontra o lixo mais próximo e sua distância"""
        nearest_trash = min(self.trash_positions, 
                        key=lambda trash: calculate_distance(agent_pos, trash))
        min_dist = calculate_distance(agent_pos, nearest_trash)
        return nearest_trash, min_dist

    def _calculate_urgency_factor(self, battery_level: float, distance: float) -> float:
        """Calcula fator de urgência baseado na bateria e distância"""
        battery_factor = 0.5 + (battery_level ** 2)  # Mais urgente com bateria cheia
        distance_factor = 1 - (distance / (self.width * 0.5))  # Mais urgente quando perto
        return battery_factor * distance_factor

    def _calculate_avoidance_penalty(self, agent_idx: int, battery_level: float, 
                                distance: float, dist_diff: float) -> float:
        """Calcula penalidade por afastamento do lixo"""
        base_penalty = Config.AWAY_FROM_TRASH_PENALTY_BASE
        
        # Ajusta multiplicador baseado na bateria
        if battery_level > 0.7:
            multiplier = Config.AWAY_FROM_TRASH_MULTIPLIER_70
        elif battery_level > 0.4:
            multiplier = Config.AWAY_FROM_TRASH_MULTIPLIER_40
        else:
            multiplier = 1.0
        
        # Ajusta pela distância (penalidade maior se estava perto)
        distance_factor = 1 + (1 - (distance / (self.width * 0.5)))
        
        return base_penalty * multiplier * distance_factor * dist_diff
    def _calculate_charging_reward(self, agent_idx: Optional[int] = None, agent_pos: Optional[List[int]] = None,battery_urgency: Optional[float] = None) -> float:
        """Recompensa por recarregar com prioridade baseada na necessidade
        
        Args:
            agent_idx: Índice do agente (None para single-agent)
            agent_pos: Posição [x,y] do agente (None para single-agent)
            battery_urgency: Urgência da bateria (0-1, None para calcular)
            
        Returns:
            Recompensa total por recarga
        """
        # Determina valores padrão para single-agent
        if agent_idx is None:
            agent_pos = self.agent_pos
            battery_level = self.battery
        else:
            battery_level = self.batteries[agent_idx]
        
        # Calcula battery_urgency se não foi fornecido
        if battery_urgency is None:
            battery_urgency = 1 - (battery_level / Config.MAX_BATTERY)
        
        charger_dist = calculate_distance(agent_pos, self.charger_pos)
        reward = 0
        
        # Dentro do carregador
        if charger_dist < Config.CHARGER_RADIUS:
            # Recarrega bateria
            charge_amount = min(2.0, Config.MAX_BATTERY - battery_level)
            if agent_idx is not None:
                self.batteries[agent_idx] = min(Config.MAX_BATTERY, self.batteries[agent_idx] + charge_amount)
            else:
                self.battery = min(Config.MAX_BATTERY, self.battery + charge_amount)
            
            # Reduz recompensa se bateria já estiver alta
            if battery_urgency < 0.4:  # Bateria > 60%
                reward += Config.CHARGER_REWARD * 0.2  # Recompensa mínima
            else:
                reward += Config.CHARGER_REWARD * (1 + battery_urgency**2)
            
            # Recompensa baseada na urgência
            base_reward = Config.CHARGER_REWARD * (1 + battery_urgency**2)
            reward += base_reward
            
        # 1. Prioridade: Recompensa por aproximação quando realmente precisa (bateria baixa)
        if battery_urgency > Config.CHARGER_HIGH_BATTERY and charger_dist < Config.CHARGER_RADIUS * 5:
            # Recompensa progressiva (mais perto = maior recompensa)
            norm_dist = charger_dist / (Config.CHARGER_RADIUS * 5)
            reward += Config.APPROACH_REWARD_FACTOR * (1 - norm_dist) * battery_urgency

        # 2. Penalidade por aproximação com bateria alta (desincentivo)
        elif battery_urgency < Config.CHARGER_HIGH_BATTERY:
            if charger_dist < Config.CHARGER_RADIUS * 5:
                # Penalidade progressiva (mais perto = maior penalidade)
                norm_dist = charger_dist / (Config.CHARGER_RADIUS * 5)
                reward -= Config.APPROACH_PENALTY_FACTOR * (1 - norm_dist) * (1 - battery_urgency)
            
            # 3. Recompensa por manter distância com bateria cheia
            elif (battery_urgency < Config.CHARGER_FULL_BATTERY and 
                charger_dist > Config.CHARGER_RADIUS * 3):
                # Recompensa por estar longe (evita ficar circulando o carregador)
                norm_dist = charger_dist / (Config.CHARGER_RADIUS * 10)
                reward += Config.DISTANCE_REWARD_FACTOR * norm_dist
        
        # Recompensa por se aproximar (quando realmente precisa)
        elif charger_dist < Config.CHARGER_RADIUS * 5 and battery_urgency > 0.5:
            approach_reward = 0.3 * (1 - charger_dist/(Config.CHARGER_RADIUS * 5))
            reward += approach_reward * battery_urgency
        
        return reward


    def _update_memory(self, agent_idx: int):
        """Atualiza memória de um agente específico"""
        decay_factor = 0.9
        self.memory_vectors[agent_idx][2::3] *= decay_factor

        visible_trash, _, _ = self.get_visible_objects()
        for trash in visible_trash:
            self._update_agent_memory(agent_idx, trash)

    def _update_agent_memory(self, agent_idx: int, trash: Tuple[int, int]):
        """Atualiza memória de um agente para um lixo específico"""
        found = False
        trash_norm = normalize_position(trash, self.width, self.height)
        
        for i in range(0, len(self.memory_vectors[agent_idx]), 3):
            mem_x, mem_y = (self.memory_vectors[agent_idx][i], 
                        self.memory_vectors[agent_idx][i+1])
            if (abs(mem_x - trash_norm[0]) < 0.05 and 
                abs(mem_y - trash_norm[1]) < 0.05):
                self.memory_vectors[agent_idx][i:i+3] = [*trash_norm, 1.0]
                found = True
                break

        if not found:
            weakest_pos = np.argmin(self.memory_vectors[agent_idx][2::3]) * 3
            self.memory_vectors[agent_idx][weakest_pos:weakest_pos+3] = [*trash_norm, 1.0]

    def render(self, episode: Optional[int] = None, max_reward: Optional[float] = None,elapsed_time: Optional[float] = None):
        """Renderiza o estado atual do jogo competitivo"""
        if not self.render_flag:
            return
            
        try:
            self.screen.fill((240, 240, 240))
            
            # Desenha carregador
            charger_rect = pygame.Rect(self.charger_pos[0]-15, self.charger_pos[1]-15, 30, 30)
            pygame.draw.rect(self.screen, (0, 200, 0), charger_rect)
            
            # Desenha todos os lixos NÃO visíveis (em marrom)
            visible_trash, _, _ = self.get_visible_objects()
            for trash in self.trash_positions:
                if trash not in visible_trash:
                    pygame.draw.circle(self.screen, (139, 69, 19), trash, 8)
            
            # Desenha os lixos visíveis (em verde)
            for trash in visible_trash:
                pygame.draw.circle(self.screen, (0, 255, 0), trash, 8)
            
            # Desenha agentes e suas informações
            for idx, agent_pos in enumerate(self.agent_positions):
                # Desenha o agente
                pygame.draw.circle(self.screen, self.agent_colors[idx], agent_pos, 12)
                
                # Desenha linhas para lixos visíveis
                for trash in visible_trash:
                    pygame.draw.line(self.screen, (0, 255, 0), agent_pos, trash, 1)
                
                # Desenha linha para carregador se visível
                charger_visible = calculate_distance(agent_pos, self.charger_pos) < 200  # Exemplo de FOV
                if charger_visible:
                    pygame.draw.line(self.screen, (255, 0, 0), agent_pos, self.charger_pos, 1)
                
                # Desenha informações do agente (pontuação e bateria)
                agent_color = self.agent_colors[idx]
                
                # Círculo identificador
                pygame.draw.circle(self.screen, agent_color, (30, 30 + idx * 30), 8)
                
                # Texto com pontuação e bateria
                agent_info = self.font.render(
                    f'Agente {idx}: {int(self.current_scores[idx])} pts | Bateria: {int(self.batteries[idx])}%',
                    True, agent_color)
                self.screen.blit(agent_info, (50, 20 + idx * 30))

            # Informações gerais
            if episode is not None and max_reward is not None:
                info_text = self.font.render(
                    f'Episódio: {episode} | Máxima: {max_reward:.1f}', 
                    True, (0, 0, 0))
                self.screen.blit(info_text, (self.width // 2 - 100, 10))

            pygame.display.flip() # Atualiza a tela
            pygame.time.delay(1) # Atraso para controle de FPS
            
        except pygame.error as e:
            logger.error(f"Erro ao renderizar: {e}")
            self.render_flag = False
class EnhancedMemory:
    """Sistema de memória que só memoriza carregadores"""
    
    def __init__(self, memory_size: int = Config.MEMORY_SIZE):
        self.memories = deque(maxlen=memory_size)
        
    def add(self, position: Tuple[float, float], object_type: str, importance: float):
        """Só adiciona à memória se for um carregador"""
        if object_type == "charger":  # Só memoriza carregadores
            self.memories.append({
                'position': position,
                'type': object_type,
                'importance': importance
            })
    
    def recall(self, current_pos: Tuple[float, float], radius: float) -> List[Dict]:
        """Retorna apenas memórias de carregadores"""
        relevant = []
        for mem in self.memories:
            dist = calculate_distance(current_pos, mem['position'])
            if dist <= radius:
                relevant.append(mem.copy())  # Retorna cópia para não alterar original
        return relevant
# ==================== PLOTTING FUNCTIONS ====================
def plot_training_progress(rewards_history: List[List[float]], 
                        battery_history: List[List[float]] = None, 
                        trash_history: List[List[int]] = None):
    """Plota gráficos do progresso do treinamento"""
    plt.figure(figsize=(15, 10))
    
    # Gráfico de recompensas
    plt.subplot(2, 1, 1)
    for i, rewards in enumerate(rewards_history):
        if rewards:  # Verifica se a lista não está vazia
            plt.plot(downsample_data(rewards), label=f'Agente {i+1}')
    plt.title('Recompensas por Episódio')
    plt.xlabel('Episódios')
    plt.ylabel('Recompensa')
    plt.legend()
    plt.grid()
    
    # Gráfico de lixo coletado (se disponível)
    if trash_history and any(trash_history):
        plt.subplot(2, 1, 2)
        for i, trash in enumerate(trash_history):
            if trash:  # Verifica se a lista não está vazia
                plt.plot(downsample_data(trash), label=f'Agente {i+1}')
        plt.title('Lixo Coletado por Episódio')
        plt.xlabel('Episódios')
        plt.ylabel('Lixo Coletado')
        plt.legend()
        plt.grid()
    
    plt.tight_layout()
    plt.savefig("training_progress.png")  # Salva a figura
    plt.show()

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    try:
        logger.info("Iniciando treinamento...")
        
        # Configurações
        config = {
            'num_agents': 10,
            'with_curiosity': False,
            'episodes': 1000,
            'render': False
        }
        
        # Seleciona modo de treinamento
        trainer_type = 'evolutionary'  # 'multi' or 'evolutionary'
        
        if trainer_type == 'multi':
            trainer = MultiAgentTrainer(
                num_agents=config['num_agents'],
                with_curiosity=config['with_curiosity'],
                render=config['render']
            )
            # Para MultiAgentTrainer
            rewards = trainer.train(episodes=config['episodes'])
        else:
            trainer = EvolutionaryTrainer(
                population_size=config['num_agents'],
                with_curiosity=config['with_curiosity'],
                render=config['render']
            )
            # Para EvolutionaryTrainer
            rewards = trainer.train(total_episodes=config['episodes'])
        
        # Plota resultados e salva atenção
        plot_training_progress([rewards])
        if hasattr(trainer, 'visualize_all_agents_attention'):
            trainer.visualize_all_agents_attention()
            trainer.visualize_attention(agent_idx=0, 
                save_path=r"C:\Users\drrod\OneDrive\Área de Trabalho\novo\polt\agent1_attention.png")
        
        logger.info("Treinamento concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro no programa principal: {e}")
    finally:
        if 'trainer' in locals() and hasattr(trainer, 'game') and trainer.game.render_flag:
            pygame.quit()
        logger.info("Programa encerrado.")
