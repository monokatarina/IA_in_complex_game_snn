import matplotlib.pyplot as plt
from typing import List

def plot_training_progress(rewards_history: List[List[float]], 
                        battery_history: List[List[float]] = None, 
                        trash_history: List[List[int]] = None):
    """Plota gráficos do progresso do treinamento

    Args:
        rewards_history: Histórico de recompensas por episódio
        battery_history: Histórico de níveis de bateria (opcional)
        trash_history: Histórico de lixo coletado (opcional)
    """
    plt.figure(figsize=(15, 10))

    # Gráfico de recompensas
    plt.subplot(2, 1, 1)
    for i, rewards in enumerate(rewards_history):
        plt.plot(rewards, label=f'Agente {i+1}')
    plt.title('Recompensas por Episódio')
    plt.xlabel('Episódios')
    plt.ylabel('Recompensa')
    plt.legend()
    plt.grid()

    # Gráfico de lixo coletado (se disponível)
    if trash_history and any(trash_history):
        plt.subplot(2, 1, 2)
        for i, trash in enumerate(trash_history):
            plt.plot(trash, label=f'Agente {i+1}')
        plt.title('Lixo Coletado por Episódio')
        plt.xlabel('Episódios')
        plt.ylabel('Lixo Coletado')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    try:
        plt.savefig("training_progress.png", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Erro ao salvar o gráfico: {e}")
    finally:
        plt.close()  # Fecha a figura para liberar memória