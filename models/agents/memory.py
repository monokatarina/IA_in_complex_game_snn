import os
from collections import deque
from typing import List, Tuple, Dict
from models.utils.helpers import calculate_distance
from models.game.config import Config

# Moved EnhancedMemory class here from trash-collector-ai.py
# This class handles memory management for agents and belongs to the Model layer.

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