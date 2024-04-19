import torch
import math
from env.oware_env_ import OwareEnv
from agents.ddqn_agent import Agent
from utils.replay_buffer import ReplayBuffer

def main():

    # Environment setup
    env = OwareEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Hyperparameters
    num_episodes = 3000
    batch_size = 64
    gamma = 0.99  # Discount factor for future rewards
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 500
    target_update = 10
    buffer_size = 10000
    min_buffer_size = 1000

    # Initialize replay buffer and agent
    replay_buffer = ReplayBuffer(buffer_size)
    agent = Agent(state_dim, action_dim, replay_buffer)

    # Variables to track progress
    steps_done = 0

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Save transition to replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state

            # Training the model
            if len(replay_buffer) > min_buffer_size:
                agent.train(batch_size, gamma)

            steps_done += 1

        # Updating the target network
        if episode % target_update == 0:
            agent.update_target_net()

        print(f'Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}')

if __name__ == '__main__':
    main()
