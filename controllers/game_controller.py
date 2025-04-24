
from models.game.core import TrashCollectionGame
from models.game.config import Config

# Initialize the game environment
def initialize_game(render: bool = True):
    """Initializes the Trash Collection Game environment."""
    return TrashCollectionGame(render=render)

# Game loop logic
def run_game(game: TrashCollectionGame, episodes: int = 100):
    """Runs the game for a specified number of episodes."""
    for episode in range(episodes):
        state = game.reset()
        done = False
        total_reward = 0

        while not done:
            # Example action selection (random for now)
            action = game.actions[4]  # Idle action
            state, reward, done = game.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")