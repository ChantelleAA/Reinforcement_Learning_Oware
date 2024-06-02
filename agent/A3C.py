import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import threading
import multiprocessing
from env import RandomAgent, GameController

class A3CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.global_model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def build_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        shared = layers.Dense(128, activation='relu')(inputs)
        shared = layers.Dense(128, activation='relu')(shared)
        actor = layers.Dense(self.action_size, activation='softmax')(shared)
        critic = layers.Dense(1)(shared)
        model = tf.keras.Model(inputs=inputs, outputs=[actor, critic])
        return model

    def train(self, env_name, num_workers):
        res_queue = multiprocessing.Queue()
        workers = [Worker(self.global_model, self.optimizer, res_queue, i, env_name, self.state_size, self.action_size) for i in range(num_workers)]

        for worker in workers:
            worker.start()

        results = []
        while True:
            reward = res_queue.get()
            if reward is not None:
                results.append(reward)
            else:
                break

        for worker in workers:
            worker.join()

        return results

class Worker(threading.Thread):
    def __init__(self, global_model, optimizer, res_queue, idx, env_name, state_size, action_size):
        super(Worker, self).__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.local_model = self.build_model()
        self.worker_idx = idx
        self.env = GameController()  # Replace with your game controller
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.res_queue = res_queue
        self.local_model.set_weights(self.global_model.get_weights())

    def build_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        shared = layers.Dense(128, activation='relu')(inputs)
        shared = layers.Dense(128, activation='relu')(shared)
        actor = layers.Dense(self.action_size, activation='softmax')(shared)
        critic = layers.Dense(1)(shared)
        model = tf.keras.Model(inputs=inputs, outputs=[actor, critic])
        return model

    def run(self):
        total_step = 1
        while True:
            current_state = self.env.reset_game()
            current_state = np.reshape(current_state, [1, self.state_size])
            done = False
            mem = []

            while not done:
                logits, value = self.local_model(tf.convert_to_tensor(current_state, dtype=tf.float32))
                policy = tf.nn.softmax(logits)
                action = np.random.choice(self.action_size, p=policy.numpy()[0])

                next_state, reward, done, _ = self.env.step(action, player=1)
                next_state = np.reshape(next_state, [1, self.state_size])
                mem.append((current_state, action, reward, next_state, done))

                if total_step % 10 == 0 or done:
                    self.update_global(mem)
                    mem = []

                current_state = next_state
                total_step += 1

            self.res_queue.put(None)

    def update_global(self, mem):
        with tf.GradientTape() as tape:
            total_loss = 0
            for state, action, reward, next_state, done in mem:
                logits, value = self.local_model(tf.convert_to_tensor(state, dtype=tf.float32))
                _, next_value = self.local_model(tf.convert_to_tensor(next_state, dtype=tf.float32))

                target = reward + (1 - done) * self.gamma * next_value[0]
                advantage = target - value[0]

                actor_loss = -tf.math.log(tf.nn.softmax(logits)[0][action]) * advantage
                critic_loss = tf.square(advantage)
                total_loss += actor_loss + critic_loss

            grads = tape.gradient(total_loss, self.local_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_weights))
            self.local_model.set_weights(self.global_model.get_weights())
