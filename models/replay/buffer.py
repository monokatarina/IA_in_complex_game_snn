import numpy as np
from typing import List, Tuple
from collections import namedtuple

# Importações do projeto
from models.utils.helpers import downsample_data

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
        if priorities.sum() == 0:
            priorities += 1e-5  # Evita divisão por zero
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