from env import DQNAgent, RandomAgent, GameController
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

INVALID_ACTION_PENALTY = -1

def zero_row_exists(state, board):
    for i in range(board.nrows):
        if np.sum(state[i,:], axis = None)==0 :
            return True
    return False

def valid_moves(state, board):
    actions = board.actions
    valid=[]
    for i in actions:
        pit_index = board.action2pit(i)
        if state[pit_index] != 0:
            if zero_row_exists(state, board) and state[pit_index]> 6-i%6:
                valid.append(i)
            elif not zero_row_exists(state, board):
                valid.append(i)
    return valid

def possible_states(player, state, board):
    actions = board.actions
    player_id = player-1
    open_moves = []
    state = np.reshape(state, (2, -1))
    for i in actions:
        pit_index = board.action2pit(i)
        if pit_index in board.player_territories[player_id]:
            if i in valid_moves(state, board) :
                open_moves.append(i)
    return open_moves

def dqn_act(state, agent, board):
    valid_actions = possible_states(agent.player_id, state, board)
    if not valid_actions:
        return None, INVALID_ACTION_PENALTY  # Return None for action and a penalty for invalid state

    if np.random.rand() <= agent.epsilon:
        return random.choice(valid_actions), 0
    else:
        act_values = agent.model.predict(state)
        act_values[0][[action for action in range(agent.action_size) if action not in valid_actions]] = float('-inf')
        return np.argmax(act_values[0]), 0

def train_agents(episodes, game_controller, save_weights_path, save_model_path):
    action_size = game_controller.action_space_size
    state_size = game_controller.state_space_size
    learning_agent = DQNAgent(state_size, action_size, 1)
    random_agent = RandomAgent(action_size, game_controller.board, 2)  # Initialize the random agent
    rewards_learner = []
    rewards_random = []

    for e in range(episodes):
        state = game_controller.reset_game()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            
            # Learning agent's turn (DQN Agent)
            action1, penalty1 = dqn_act(state, learning_agent, game_controller.board)

            next_state, reward1, done, info = game_controller.step(action1, player=1)
            next_state = np.reshape(next_state, [1, state_size])
            learning_agent.remember(state, action1, reward1 + penalty1, next_state, done)
            state = next_state
            rewards_learner.append(reward1)

            if done:
                print(f"Episode: {e+1}/{episodes}, Epsilon: {learning_agent.epsilon:.2f}")
                break

            action2, penalty2 = random_agent.act(state, game_controller.board)
            next_state, reward2, done, info = game_controller.step(action2, player=2)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state  # No memory or learning for the random agent
            rewards_random.append(reward2)

            # Only train the learning agent
            if len(learning_agent.memory) > 32:
                learning_agent.replay(32)

            if done:
                print(f"Episode: {e+1}/{episodes}, Epsilon: {learning_agent.epsilon:.2f}")
                break

        learning_agent.save(save_weights_path.format(e+1))
        learning_agent.save_model(save_model_path.format(e+1))

    print("Training completed and model weights saved for the learning agent.")
    plt.plot(rewards_learner, "r" , label="DQN agent")
    plt.plot(rewards_random, "b" , label="Random agent")
    plt.legend()
    plt.show()
game_controller = GameController()
train_agents(200, game_controller, './saved_weights/saved_weights_epoch_{}.h5', './saved_weights/saved_model_epoch_{}.json')
