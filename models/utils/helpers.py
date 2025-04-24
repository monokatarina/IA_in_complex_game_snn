from typing import List, Tuple
import math
import numpy as np

# Moved helper functions here from trash-collector-ai.py
# These functions provide utility methods and belong to the Model layer.

def normalize_position(pos: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
    """Normaliza coordenadas para o intervalo [0,1]
    
    Args:
        pos: Tupla com coordenadas (x, y)
        width: Largura do ambiente
        height: Altura do ambiente
        
    Returns:
        Tupla com coordenadas normalizadas
    """
    if width <= 0 or height <= 0:
        raise ValueError("Largura e altura devem ser maiores que zero.")
    return (pos[0] / width, pos[1] / height)

def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calcula distância euclidiana entre dois pontos
    
    Args:
        pos1: Primeiro ponto (x, y)
        pos2: Segundo ponto (x, y)
        
    Returns:
        Distância entre os pontos
    """
    if not all(isinstance(coord, (int, float)) for coord in (*pos1, *pos2)):
        raise TypeError("As coordenadas devem ser números.")
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