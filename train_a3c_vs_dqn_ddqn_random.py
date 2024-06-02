from env import DQNAgent, DDQNAgent, RandomAgent, GameController, A3CAgent
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def train_a3c(env_name, num_workers, num_episodes):
    game_controller = GameController()
    state_size = game_controller.state_space_size
    action_size = game_controller.action_space_size
    a3c_agent = A3CAgent(state_size, action_size)
    results = a3c_agent.train(env_name, num_workers)
    
    plt.plot(results)
    plt.ylabel('Rewards')
    plt.xlabel('Episode')
    plt.show()

def train_against_agents(num_episodes, agent_type='dqn'):
    game_controller = GameController()
    action_size = game_controller.action_space_size
    state_size = game_controller.state_space_size

    if agent_type == 'dqn':
        opponent_agent = DQNAgent(state_size, action_size, 2)
    elif agent_type == 'ddqn':
        opponent_agent = DDQNAgent(state_size, action_size, 2)
    else:
        opponent_agent = RandomAgent(action_size, game_controller.board, 2)
        
    a3c_agent = A3CAgent(state_size, action_size)
    
    rewards_a3c = []
    rewards_opponent = []

    for e in range(num_episodes):
        state = game_controller.reset_game()
        state = np.reshape(state, [1, state_size])
        done = False
        episode_reward_a3c = 0
        episode_reward_opponent = 0

        while not done:
            logits, value = a3c_agent.global_model(tf.convert_to_tensor(state, dtype=tf.float32))
            policy = tf.nn.softmax(logits)
            action = np.random.choice(action_size, p=policy.numpy()[0])

            next_state, reward, done, _ = game_controller.step(action, player=1)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            episode_reward_a3c += reward
            rewards_a3c.append(reward)

            if done:
                break

            action_opponent = opponent_agent.act(state)
            next_state, reward_opponent, done, _ = game_controller.step(action_opponent, player=2)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            episode_reward_opponent += reward_opponent
            rewards_opponent.append(reward_opponent)

            if done:
                break

        print(f"Episode: {e+1}/{num_episodes}, A3C Reward: {episode_reward_a3c}, Opponent Reward: {episode_reward_opponent}")

    np.savetxt(f"rewards_a3c_vs_{agent_type}.csv", rewards_a3c)
    np.savetxt(f"rewards_opponent_{agent_type}.csv", rewards_opponent)    

    plt.plot(rewards_a3c, "r" , label="A3C agent")
    plt.plot(rewards_opponent, "b" , label=f"{agent_type} agent")
    plt.legend()
    plt.show()

# Train A3C against random agent
train_a3c('GameEnv', num_workers=4, num_episodes=30)

# Train A3C against DQN agent
train_against_agents(num_episodes=30, agent_type='dqn')

# Train A3C against DDQN agent
train_against_agents(num_episodes=30, agent_type='ddqn')

# Train A3C against random agent
train_against_agents(num_episodes=30, agent_type='random')
