import pygame
from typing import Optional

# Render the current state of the game
def render(screen, agent_color, agent_pos, charger_pos, trash_positions, collected_trash, episode=None, reward=None):
    """Renderiza o estado atual do jogo

    Args:
        screen: Pygame screen object
        agent_color: Color of the agent
        agent_pos: Position of the agent
        charger_pos: Position of the charger
        trash_positions: List of trash positions
        collected_trash: Number of collected trash
        episode: Current episode number
        reward: Accumulated reward
    """
    try:
        if not pygame.get_init():
            raise pygame.error("Pygame não foi inicializado corretamente.")

        # Clear the screen
        screen.fill((240, 240, 240))

        # Draw the charger
        pygame.draw.rect(screen, (0, 200, 0), (charger_pos[0]-15, charger_pos[1]-15, 30, 30))

        # Draw all trash
        for trash in trash_positions:
            pygame.draw.circle(screen, (255, 0, 0), trash, 8)

        # Draw the agent
        pygame.draw.circle(screen, agent_color, agent_pos, 12)

        # Highlight if the agent is charging
        charger_dist = ((agent_pos[0] - charger_pos[0])**2 + (agent_pos[1] - charger_pos[1])**2)**0.5
        if charger_dist < 25:  # Assuming CHARGER_RADIUS = 25
            pygame.draw.circle(screen, (255, 255, 0), agent_pos, 15, 2)

        # Draw text information
        font = pygame.font.SysFont(None, 24)
        trash_text = font.render(f'Lixo: {collected_trash}/10', True, (0, 0, 0))
        screen.blit(trash_text, (10, 40))

        if episode is not None and reward is not None:
            info_text = font.render(f'Episódio: {episode} | Recompensa: {reward:.1f}', True, (0, 0, 0))
            screen.blit(info_text, (10, 70))

        pygame.display.flip()

    except pygame.error as e:
        print(f"Erro de renderização: {e}")
        pygame.quit()  # Garante que o Pygame seja encerrado corretamente