# Batch size for training the DQN model
BATCH_SIZE = 8

# Discount factor for future rewards in the Q-learning algorithm
GAMMA = 0.999

# Starting value for the exploration rate (epsilon) in the epsilon-greedy policy
EPS_START = 1

# Ending value for the exploration rate (epsilon) in the epsilon-greedy policy
EPS_END = 0.05

# Decay rate for the exploration rate (epsilon) in the epsilon-greedy policy
EPS_DECAY = 200000

# Soft update coefficient for updating the target network in the DQN algorithm
TAU = 0.002

# Starting learning rate for the optimizer
LR_START = 0.003

# Ending learning rate for the optimizer
LR_END = 0.0002

# Decay rate for the learning rate
LR_DECAY = 2000000

# Player ID DQN agent
PLAYER_NUM_DQN = 1

# Player ID Random agent
PLAYER_NUM_RANDOM = 2

# Player ID Human agent
PLAYER_NUM_HUMAN = 2

# Rewards
INVALID_ACTION_PENALTY = -10 
