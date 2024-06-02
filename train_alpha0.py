from env import AlphaZeroAgent, mcts_policy, DDQNAgent, DQNAgent, A3CAgent, RandomAgent, GameController
import tensorflow as tf
import numpy as np
import datetime 
import matplotlib.pyplot as plt

log_dir = "logs1/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')
file_writer = tf.summary.create_file_writer(log_dir)


def train_alphazero(episodes, game_controller, save_weights_path, save_model_path, opponent_agent):
    action_size = game_controller.action_space_size
    state_size = game_controller.state_space_size
    learning_agent = AlphaZeroAgent(state_size, action_size, 1)
    
    rewards_learner = []
    rewards_opponent = []

    # Training loop
    batch_size = 32 

    for e in range(episodes):
        state = game_controller.reset_game()
        state = np.reshape(state, [1, state_size])
        done = False
        episode_reward_learner = 0
        episode_reward_opponent = 0

        while not done:
            policy = mcts_policy(state, learning_agent)
            action = np.random.choice(action_size, p=policy)
            next_state, reward, done, _ = game_controller.step(action, player=1)
            next_state = np.reshape(next_state, [1, state_size])
            learning_agent.remember(state, policy, reward)
            state = next_state
            episode_reward_learner += reward

            if done:
                break

            action_opponent = opponent_agent.act(state)
            next_state, reward_opponent, done, _ = game_controller.step(action_opponent, player=2)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            episode_reward_opponent += reward_opponent

            if done:
                break

        print(f"Episode: {e+1}/{episodes}, AlphaZero Reward: {episode_reward_learner}, Opponent Reward: {episode_reward_opponent}")

        rewards_learner.append(episode_reward_learner)
        rewards_opponent.append(episode_reward_opponent)

        if len(learning_agent.memory) > batch_size:
            learning_agent.replay(batch_size)

        learning_agent.save(save_weights_path.format(e+1))

        # Log the episode reward
        with file_writer.as_default():
            tf.summary.scalar('Episode Reward', episode_reward_learner, step=e)
            tf.summary.flush()

    learning_agent.save_model(save_model_path)
    np.savetxt("rewards_learner.csv", rewards_learner)
    np.savetxt("rewards_opponent.csv", rewards_opponent)    

    print("Training completed and model weights saved for the learning agent.")

    plt.subplot(1, 2, 1)
    plt.plot(rewards_learner, "r", label="AlphaZero agent")
    plt.plot(rewards_opponent, "b", label="Opponent agent")

    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(np.array(rewards_learner)), "r", label="cumsum AlphaZero")
    plt.plot(np.cumsum(np.array(rewards_opponent)), "b", label="cumsum Opponent")

    plt.legend()
    plt.show()

# Initialize the GameController
game_controller = GameController()
state_size = game_controller.state_space_size
action_size = game_controller.action_space_size

# Train AlphaZero against Random agent
random_agent = RandomAgent(action_size, game_controller.board, 2)
train_alphazero(300, game_controller, './saved_weights_alphazero_random/E{}.weights.h5', './saved_weights_alphazero_random/model_params{}.json', random_agent)

# Train AlphaZero against DQN agent
dqn_agent = DQNAgent(state_size, action_size, 2)
train_alphazero(300, game_controller, './saved_weights_alphazero_dqn/E{}.weights.h5', './saved_weights_alphazero_dqn/model_params{}.json', dqn_agent)

# Train AlphaZero against DDQN agent
ddqn_agent = DDQNAgent(state_size, action_size, 2)
train_alphazero(300, game_controller, './saved_weights_alphazero_ddqn/E{}.weights.h5', './saved_weights_alphazero_ddqn/model_params{}.json', ddqn_agent)

# Train AlphaZero against A3C agent
a3c_agent = A3CAgent(state_size, action_size)
train_alphazero(300, game_controller, './saved_weights_alphazero_a3c/E{}.weights.h5', './saved_weights_alphazero_a3c/model_params{}.json', a3c_agent)