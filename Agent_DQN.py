import random
from sys import getsizeof

import numpy as np

import QNeuralNetwork as NN


class GameAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, batch_size=32,
                 buffer_max_size=10000, save_name='dqn', learning_time=100000, FREEZE_WEIGHTS=True,
                 DOUBLE_NETWORK=True, PRIORITIZED_XP_REPLAY=True, alpha=0.6, beta=0.4, freeze_steps=5000):
        self.freeze_steps = freeze_steps
        self.FREEZE_WEIGHTS = FREEZE_WEIGHTS
        self.DOUBLE_NETWORK = DOUBLE_NETWORK
        self.PRIORITIZED_XP_REPLAY = PRIORITIZED_XP_REPLAY
        self.learning_time = learning_time
        self.save_name = save_name
        self.buffer_max_size = buffer_max_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.indexes = np.arange(self.buffer_max_size)
        self.batch_indexes = np.arange(self.batch_size)
        if self.PRIORITIZED_XP_REPLAY:
            self.sum_prob = 0
            self.max_prob = 1
            self.experience_prob = np.zeros(self.buffer_max_size)
            self.alpha = alpha
            self.beta = beta

        print 'Online-network initializing...'
        self.online_network = NN.QNeuralNetwork(state_dim, action_dim, name='online', batch_size=batch_size)

        if self.FREEZE_WEIGHTS:
            print 'Frozen-network initializing...'
            self.frozen_network = NN.QNeuralNetwork(state_dim, action_dim, name='offline', batch_size=batch_size)

        try:
            self.online_network.model.load_weights('{}.h5'.format(self.save_name))
            if self.FREEZE_WEIGHTS:
                self.frozen_network.model.load_weights('{}.h5'.format(self.save_name))
            print "loading from {}.h5".format(self.save_name)
        except:
            print "Training a new model"

        print 'Networks initialized.'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_step = 0
        self.experience_state = np.zeros((self.buffer_max_size,) + self.state_dim, dtype='float32') * 1.0
        print getsizeof(self.experience_state)
        self.experience_action = np.zeros(self.buffer_max_size) * 1.0
        self.experience_reward = np.zeros(self.buffer_max_size, dtype='float32') * 1.0
        self.experience_done = np.zeros(self.buffer_max_size) * 1.0

        self.buffer_size = 0

    def greedy_action(self, state):
        return np.argmax(self.online_network.get_output(state))

    def e_greedy_action(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return self.greedy_action(state)

    def memorize(self, state, action, reward, terminal):
        if self.time_step % self.freeze_steps == 0:
            print 'Backing up...'
            self.online_network.model.save_weights('{}.h5'.format(self.save_name), True)
            self.online_network.model.load_weights('{}.h5'.format(self.save_name))
            if self.FREEZE_WEIGHTS:
                self.frozen_network.model.load_weights('{}.h5'.format(self.save_name))
            print 'Backed up!'
        self.time_step += 1

        self.experience_state[self.time_step % self.buffer_max_size] = state
        self.experience_reward[self.time_step % self.buffer_max_size] = reward
        self.experience_action[self.time_step % self.buffer_max_size] = action
        self.experience_done[self.time_step % self.buffer_max_size] = terminal

        if self.PRIORITIZED_XP_REPLAY:
            self.sum_prob -= self.experience_prob[self.time_step % self.buffer_max_size]
            self.experience_prob[self.time_step % self.buffer_max_size] = 0
        self.buffer_size += int(self.buffer_size < self.buffer_max_size)

        if self.time_step < self.learning_time:
            self.epsilon -= (1 - 0.1) / self.learning_time

        if self.time_step > self.buffer_max_size/2:
            self.train_step()

        if self.PRIORITIZED_XP_REPLAY:
            self.sum_prob += self.max_prob
            self.experience_prob[self.time_step % self.buffer_max_size] = self.max_prob

    def train_step(self):
        if self.PRIORITIZED_XP_REPLAY:
            probs = self.experience_prob / self.sum_prob
            indexes_batch = np.random.choice(self.indexes, size=self.batch_size, p=probs)
        else:
            indexes_batch = np.random.randint(self.buffer_max_size, size=self.batch_size)

        state_batch = self.experience_state[indexes_batch]
        action_batch = self.experience_action[indexes_batch]
        reward_batch = self.experience_reward[indexes_batch]
        next_state_batch = self.experience_state[(indexes_batch + 1) % self.buffer_max_size]
        done_batch = self.experience_done[indexes_batch]
        prob_batch = self.experience_prob[indexes_batch]

        if self.FREEZE_WEIGHTS:
            if self.DOUBLE_NETWORK:
                q_argmax_online = np.argmax(self.online_network.get_output(next_state_batch), axis=1)
                output_frozen = self.frozen_network.get_output(next_state_batch)
                q_max_batch = output_frozen[self.batch_indexes, q_argmax_online]
            else:
                q_max_batch = np.max(self.frozen_network.get_output(next_state_batch), axis=1)
        else:
            q_max_batch = np.max(self.online_network.get_output(next_state_batch), axis=1)

        y_batch = (reward_batch + (1 - done_batch) * self.gamma * q_max_batch).reshape((self.batch_size, 1))
        action_batch = action_batch.reshape((self.batch_size, 1))

        state_batch = state_batch.reshape((self.batch_size,) + self.state_dim)

        if self.PRIORITIZED_XP_REPLAY:
            weights_batch = (self.buffer_size * prob_batch) ** self.beta
            weights_batch /= weights_batch.max()
            cost, error = self.online_network.train_step(y_batch, state_batch, action_batch, weights_batch)
            for i in range(self.batch_size):
                self.sum_prob = self.sum_prob + abs(error[i]) ** self.alpha - self.experience_prob[indexes_batch[i]]
                self.experience_prob[indexes_batch[i]] = abs(error[i]) ** self.alpha
        else:
            cost = self.online_network.train_step(y_batch, state_batch, action_batch)[0]

        if self.time_step % 500 == 0:
            print cost
