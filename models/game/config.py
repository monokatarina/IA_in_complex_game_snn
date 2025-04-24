class RewardConfig:
    """Configurações hierárquicas de recompensa para o agente"""
    class Primary:
        TRASH_COLLECTED = 400.0
        COMPLETION_BONUS = 250.0
        BATTERY_PENALTY = -100.0

    class Battery:
        BONUS = 15.0
        BONUS_MULTIPLIER = 1.7
        CHARGER_REWARD = 1.0
        HIGH_LEVEL = 0.6
        LOW_LEVEL = 0.3
        CRITICAL_LEVEL = 0.1
        IDLE_AT_CHARGER = -3.0
        HIGH_BATTERY_PENALTY = -2.0

    class Behavior:
        MOVEMENT_PENALTY = -0.05
        IDLE_WITH_URGENT_TASKS = -2.5
        APPROACH_TRASH = 1.0
        APPROACH_CHARGER = 2.0
        AWAY_FROM_CHARGER = 3.0
        APPROACH_PENALTY_FACTOR = 0.3
        DISTANCE_REWARD_FACTOR = 0.2
        APPROACH_REWARD_FACTOR = 0.5

    class Avoidance:
        TRASH_BASE = -1
        MULTIPLIER_40 = 1.5
        MULTIPLIER_70 = 2.0
        HIGH_BATTERY_MULTIPLIER = 2.0

    class ChargerThresholds:
        HIGH_BATTERY = 0.5
        FULL_BATTERY = 0.15

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
    TIME_LIMIT = 300  # Limite de tempo para cada episódio (em segundos)

    # Configurações da rede neural
    MEMORY_SIZE = 10
    STATE_SIZE = 22 + MEMORY_SIZE * 4

    # Configurações de treinamento
    BATCH_SIZE = 64
    GAMMA = 0.95
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995
    TARGET_UPDATE_FREQ = 300
    LEARNING_RATE = 0.0005

    # Recompensas configuradas
    TRASH_COLLECTED_REWARD = RewardConfig.Primary.TRASH_COLLECTED
    BATTERY_BONUS = RewardConfig.Battery.BONUS
    BATTERY_BONUS_MULTIPLIER = RewardConfig.Battery.BONUS_MULTIPLIER
    CHARGER_REWARD = RewardConfig.Battery.CHARGER_REWARD
    BATTERY_PENALTY = RewardConfig.Primary.BATTERY_PENALTY
    COMPLETION_BONUS = RewardConfig.Primary.COMPLETION_BONUS
    MOVEMENT_PENALTY = RewardConfig.Behavior.MOVEMENT_PENALTY
    IDLE_AT_CHARGER_PENALTY = RewardConfig.Battery.IDLE_AT_CHARGER
    IDLE_WITH_URGENT_TASKS_PENALTY = RewardConfig.Behavior.IDLE_WITH_URGENT_TASKS
    APPROACH_TRASH_REWARD = RewardConfig.Behavior.APPROACH_TRASH
    APPROACH_CHARGER_REWARD = RewardConfig.Behavior.APPROACH_CHARGER
    HIGH_BATTERY_PENALTY = RewardConfig.Battery.HIGH_BATTERY_PENALTY
    APPROACH_CHARGER_PENALTY = RewardConfig.Behavior.APPROACH_PENALTY_FACTOR
    AWAY_FROM_CHARGER_REWARD = RewardConfig.Behavior.AWAY_FROM_CHARGER
    AWAY_FROM_TRASH_PENALTY = RewardConfig.Avoidance.TRASH_BASE
    HIGH_BATTERY_AVOIDANCE_MULTIPLIER = RewardConfig.Avoidance.HIGH_BATTERY_MULTIPLIER
    CHARGER_HIGH_BATTERY = RewardConfig.ChargerThresholds.HIGH_BATTERY
    CHARGER_FULL_BATTERY = RewardConfig.ChargerThresholds.FULL_BATTERY
    APPROACH_PENALTY_FACTOR = RewardConfig.Behavior.APPROACH_PENALTY_FACTOR
    DISTANCE_REWARD_FACTOR = RewardConfig.Behavior.DISTANCE_REWARD_FACTOR
    APPROACH_REWARD_FACTOR = RewardConfig.Behavior.APPROACH_REWARD_FACTOR
    AWAY_FROM_TRASH_PENALTY_BASE = RewardConfig.Avoidance.TRASH_BASE
    AWAY_FROM_TRASH_MULTIPLIER_40 = RewardConfig.Avoidance.MULTIPLIER_40
    AWAY_FROM_TRASH_MULTIPLIER_70 = RewardConfig.Avoidance.MULTIPLIER_70