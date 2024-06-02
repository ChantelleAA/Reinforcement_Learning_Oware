from env import DQNAgent, RandomAgent, GameController
import numpy as np
import random
import datetime
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore')

INVALID_ACTION_PENALTY = -1

log_dir = "logs1/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')
file_writer = tf.summary.create_file_writer(log_dir)

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
    # print(f"Valid actions: {valid_actions}")
    if not valid_actions:
        return None, INVALID_ACTION_PENALTY
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
    trained_agent = DQNAgent(state_size, action_size, 2)
    trained_agent.model.load_weights('./saved_weights1/saved_weights_epoch_23.h5') 

    for e in range(episodes):
        state = game_controller.reset_game()
        state = np.reshape(state, [1, state_size])
        done = False
        episode_reward = 0
        episode_reward1 =0 

        while not done:
            
            # Learning agent's turn (DQN Agent)
            action1, penalty1 = dqn_act(state, learning_agent, game_controller.board)
            next_state, reward1, done, info = game_controller.step(action1, player=1)
            next_state = np.reshape(next_state, [1, state_size])
            learning_agent.remember(state, action1, reward1 + penalty1, next_state, done)
            state = next_state
            episode_reward += reward1

            if done:
                print(f"Episode: {e+1}/{episodes}, Epsilon: {learning_agent.epsilon:.2f}")
                break

            act_values = trained_agent.model.predict(state)
            action2 = np.argmax(act_values[0])
            next_state, reward2, done, info = game_controller.step(action2, player=2)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state  # Implement policy for the trained_agent
            episode_reward1 += reward2

            # Only train the learning agent
            if len(learning_agent.memory) > 32:
                learning_agent.replay(32)

            if done:
                print(f"Episode: {e+1}/{episodes}, Epsilon: {learning_agent.epsilon:.2f}")
                break

        learning_agent.save(save_weights_path.format(e+1))
        learning_agent.save_model(save_model_path.format(e+1))

        # Log the episode reward
        with file_writer.as_default():
            tf.summary.scalar('Episode Reward Learning Agent', episode_reward, step=e)
            tf.summary.scalar('Episode Reward Agent 23 ', episode_reward1, step=e)
            tf.summary.flush()

    print("Training completed and model weights saved for the learning agent.")

game_controller = GameController()
train_agents(15, game_controller, './saved_weights1/saved_weights_epoch_{}.h5', './saved_weights1/saved_model_epoch_{}.json')
