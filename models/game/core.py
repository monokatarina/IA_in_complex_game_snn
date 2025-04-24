import os
import math
import numpy as np
import random
import pygame
from typing import List, Tuple, Dict, Optional

# Importações do projeto
from models.utils.helpers import normalize_position, calculate_distance
from models.agents.memory import EnhancedMemory
from models.game.config import Config

# Configuração de logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('game.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Moved TrashCollectionGame and CompetitiveTrashGame classes here from trash-collector-ai.py
# These classes represent the core game logic and belong to the Model layer.


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
        try:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Trash Collection Game")
        except pygame.error as e:
            logger.error(f"Erro ao inicializar o Pygame: {e}")
            self.render_flag = False
        
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
        return next(
            (i for i in range(len(Config.SECTOR_ANGLES) - 1) 
             if Config.SECTOR_ANGLES[i] <= angle < Config.SECTOR_ANGLES[i + 1]), 
            0
        )

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