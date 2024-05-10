# from env import DQNAgent
# from env import GameController
# import numpy as np
# import random

# import warnings
# warnings.filterwarnings('ignore')

# INVALID_ACTION_PENALTY = -10  # Define a penalty for invalid actions

# def possible_states(state, board_indicies, agent):
#     valid_actions = []
#     state = np.reshape(state, (2, -1))
#     for action in range(agent.action_size):
#         pit_index = board_indicies[action]
#         print(f"For action {action}, {state[pit_index]=}")
#         if state[pit_index] != 0.0:
#             valid_actions.append(action)
#     if not valid_actions:
#         raise ValueError("No valid actions available for this current state")
#     return valid_actions

# # def act(state, board, game_state, agent):
# def act(state, board, agent):
#     valid_actions = possible_states(state, board, agent)
#     print(f"Players possible actions are: {valid_actions}")
#     if not valid_actions:
#         print("No valid actions available for the current state.")
#         return None
#         # raise ValueError("No valid actions available for the current state.")
        
#     if np.random.rand() <= agent.epsilon:
        
#         return random.choice(valid_actions)
#     else:
#         act_values = agent.model.predict(state)

#         # Filter out invalid actions by setting their Q-values to a large negative number
#         act_values[0][[action for action in range(agent.action_size) if action not in valid_actions]] = float('-inf')
#         return np.argmax(act_values[0])

# def train_dqn(episodes, game_controller, save_path):
#     action_size = game_controller.action_space_size
#     state_size = game_controller.state_space_size

#     agent = DQNAgent(state_size=12, action_size=12) 

#     for e in range(episodes):

#         state = game_controller.reset_game()
#         state = np.reshape(state, [1, state_size])
#         done = False

#         for _ in range(10):
#             board = game_controller.board_indicies

#             valid_actions = possible_states(state, board, agent)
            
#             action = act(state, board, agent)
            
#             # Check if the chosen action is invalid and apply penalty
#             if action not in valid_actions or action == None:
#                 print(f"\n \n INVALID ACTION RESTART \n\n")
#                 reward = INVALID_ACTION_PENALTY
#                 done = True  
#                 break 

#             next_state, reward, done, info = game_controller.step(action)

#             next_state = np.reshape(next_state, [1, state_size])
   
#             agent.remember(state, action, reward, next_state, done)
            
#             state = next_state

#             if done:
#                 print('END OF GAME')
#                 print(f"episode: {e+1}/{episodes}, score: {game_controller.get_score()}, e: {agent.epsilon:.2}")
#                 break
            
#             if len(agent.memory) > 32:
#                 agent.replay(32)

#         agent.model.save_weights(save_path.format(e+1))
    
#     print("Training completed and model weights saved.")

# game_controller = GameController() 
# train_dqn(10, game_controller, 'saved_weights_epoch_{}.h5')


from env import DQNAgent, RandomAgent, GameController
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

INVALID_ACTION_PENALTY = -10

def possible_states(state, board_indicies, agent):
    valid_actions = []
    state = np.reshape(state, (2, -1))
    for action in range(agent.action_size):
        pit_index = board_indicies[action]
        # print(f"For action {action}, {state[pit_index]=}")
        if state[pit_index] != 0.0:
            valid_actions.append(action)
    # if not valid_actions:
    #     raise ValueError("No valid actions available for this current state")
    return valid_actions

 
def dqn_act(state, board_indices, agent):
    valid_actions = possible_states(state, board_indices, agent)
    if not valid_actions:
        return None, INVALID_ACTION_PENALTY  # Return None for action and a penalty for invalid state

    if np.random.rand() <= agent.epsilon:
        return random.choice(valid_actions), 0
    else:
        act_values = agent.model.predict(state)
        act_values[0][[action for action in range(agent.action_size) if action not in valid_actions]] = float('-inf')
        return np.argmax(act_values[0]), 0

def train_agents(episodes, game_controller, save_path):
    action_size = game_controller.action_space_size
    state_size = game_controller.state_space_size
    learning_agent = DQNAgent(state_size, action_size, 0)
    random_agent = RandomAgent(action_size)  # Initialize the random agent

    for e in range(episodes):
        state = game_controller.reset_game()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            board_indices = game_controller.board.board_indices
            
            # Learning agent's turn (DQN Agent)
            action1, penalty1 = dqn_act(state, board_indices, learning_agent)
            if action1 is None:
                reward1 = penalty1
                print("No valid actions available; ending round.")
                done = True
                continue
            else:
                next_state, reward1, done, info = game_controller.step(action1, player=1)
                next_state = np.reshape(next_state, [1, state_size])
                learning_agent.remember(state, action1, reward1 + penalty1, next_state, done)
                state = next_state

            # Random agent's turn
            if not done:
                action2, penalty2 = random_agent.act(state, board_indices)
                if action2 is None:
                    reward2 = penalty2
                    done = True
                else:
                    next_state, reward2, done, info = game_controller.step(action2, player=2)
                    next_state = np.reshape(next_state, [1, state_size])
                    state = next_state  # No memory or learning for the random agent

            # Only train the learning agent
            if len(learning_agent.memory) > 32:
                learning_agent.replay(32)

            if done:
                print(f"Episode: {e+1}/{episodes}, Epsilon: {learning_agent.epsilon:.2f}")

        learning_agent.model.save_weights(save_path.format(e+1))

    print("Training completed and model weights saved for the learning agent.")

game_controller = GameController()
train_agents(2000, game_controller, './saved_weights/saved_weights_epoch_{}.h5')
