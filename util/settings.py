# Q-Learning Agent Settings
EPSILON = 1
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.05

# Game Settings
WIN_REWARD = 3
TIE_REWARD = 1

# OpenAI Gym / Ray Agent Settings
NUM_HIDDEN = 128
BAD_ACTION_REWARD = -100
OPPONENT = 'random'

# Both Q-learning and OpenAI / StableBaselines agent settings
TRAINING_GAMES = 50000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.1