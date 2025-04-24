import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controllers.training_controller import MultiAgentTrainer, EvolutionaryTrainer
from views.pygame_view import render
from views.plots import plot_training_progress

if __name__ == "__main__":
    try:
        print("Starting training...")

        # Configuration
        config = {
            'num_agents': 5,
            'with_curiosity': False,
            'episodes': 1000,
            'render': True
        }

        # Select training mode
        trainer_type = 'evolutionary'  # 'multi' or 'evolutionary'

        if trainer_type == 'multi':
            trainer = MultiAgentTrainer(
                num_agents=config['num_agents'],
                with_curiosity=config['with_curiosity'],
                render=config['render']
            )
        else:
            trainer = EvolutionaryTrainer(
                population_size=config['num_agents'],
                with_curiosity=config['with_curiosity'],
                render=config['render']
            )

        # Train the model
        rewards = trainer.train(total_episodes=config['episodes'])

        # Plot results
        plot_training_progress([rewards])

        print("Training completed successfully!")

    except KeyboardInterrupt:
        print("Treinamento interrompido pelo usuário.")
    except Exception as e:
        print(f"Erro no programa principal: {e}")
        raise  # Relevanta a exceção para facilitar a depuração
    finally:
        print("Programa encerrado.")