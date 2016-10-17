import numpy as np
import QNeuralNetwork as NN
import os


class GameAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, batch_size=32,
                 buffer_max_size=10000, save_name='dqn', learning_time=10000, n_observe=100000,
                 DOUBLE_NETWORK=False, PRIORITIZED_XP_REPLAY=False, DUELING_ARCHITECTURE=False, alpha=0.6, beta=0.4,
                 backup_steps=5000,
                 learning_rate=1., debug_steps=500, train_every_steps=4):
        """ Initialize agent.
            Args:
              state_dim: dimensionality of space of states
              action_dim: dimensionality of space of actions
              batch_size: size of minibatch for network's train
              buffer_max_size: maximum size of experience buffer
              save_name: Name of file where agent saves the weights of the Q-network
              learning_time: From first time-step to learning_time time-step agent reduces e from 1 to 0.1
              DOUBLE_NETWORK: double q-learning activation
              PRIORITIZED_XP_REPLAY: Prioritized xp-replay activation
              DUELING_ARCHITECTURE:  dueling network architecture activation
              alpha and beta: Parameters of prioritized xp-replay described above
              backup_steps: Parameter described above
              gamma: discount factor of reward
              learning_rate: learning rate of optimizer of networks
              debug_steps: How often agent prints cost of batch
              train_every_steps: how often agent makes a train step
        """
        # Assign agent features
        self.backup_steps = backup_steps
        self.DOUBLE_NETWORK = DOUBLE_NETWORK
        self.PRIORITIZED_XP_REPLAY = PRIORITIZED_XP_REPLAY
        self.learning_time = learning_time
        self.save_name = save_name
        self.buffer_max_size = buffer_max_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.end_epsilon = 0.1
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.debug_steps = debug_steps
        self.train_every_steps = train_every_steps
        self.n_observe = n_observe
        # Create some supporting variables
        self.indexes = np.arange(self.buffer_max_size)
        self.batch_indexes = np.arange(self.batch_size)
        self.time_step = 0
        self.buffer_size = 0
        self.delta_eps = (self.epsilon - self.end_epsilon) / learning_time
        self.last_q = 0
        if self.PRIORITIZED_XP_REPLAY:
            # It is the sum of the priorities of all transitions.
            # Agent maintains it just because it's inefficiently to compute that big sum at every step
            self.sum_prior = 0
            # When agent meets new transition we set maximum priority to it because we want to try to optimize it
            # Agent maintains it as well
            self.max_prior = 1
            # Here are two prioritized xp-replay parameters
            self.alpha = alpha
            self.beta = beta

        # Networks initializing
        print 'Online-network initializing...'
        self.online_network = NN.QNeuralNetwork(state_dim, action_dim, batch_size=batch_size,
                                                learning_rate=learning_rate, DUELING_ARCHITECTURE=DUELING_ARCHITECTURE)
        print 'Frozen-network initializing...'
        self.frozen_network = NN.QNeuralNetwork(state_dim, action_dim, batch_size=batch_size,
                                                learning_rate=learning_rate, DUELING_ARCHITECTURE=DUELING_ARCHITECTURE)

        # Try to load weights if we made an agent for our task before

        try:
            print "Try to load networks from files..."
            self.online_network.model.load_weights(self.save_name + '/weights.h5')
            self.frozen_network.model.load_weights(self.save_name + '/weights.h5')
            self.time_step = np.load(self.save_name + '/step.npy')[0]
            print "Networks are loaded from {}.h5".format(self.save_name)
        except:
            print "Training a new model"

        print 'Networks initialized.'
        self.epsilon = 1 - (1 - self.end_epsilon) * min(1, max(0, float(
            self.time_step - self.n_observe)) / self.learning_time)
        # Initializing of experience buffer
        self.experience_state = np.zeros((self.buffer_max_size,) + self.state_dim, dtype=np.uint8) + 1
        self.experience_action = np.zeros(self.buffer_max_size) * 1.0
        self.experience_reward = np.zeros(self.buffer_max_size) * 1.0
        self.experience_done = np.zeros(self.buffer_max_size) * 1.0
        self.buffer_last = 0
        if self.PRIORITIZED_XP_REPLAY:
            self.experience_prob = np.zeros(self.buffer_max_size) * 1.0

    def greedy_action(self, state):
        # This is a function for environment-agent interaction
        act = np.argmax(self.online_network.get_output(state)[:self.action_dim])
        # print self.online_network.get_output(state)[:self.action_dim]
        return act

    def e_greedy_action(self, state, epsilon=None):
        # This is a function for environment-agent interaction
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_dim)
        else:
            qs = self.online_network.get_output(state)[:self.action_dim]
            if epsilon == 0.04:
                print qs
            return np.argmax(qs)

    def action(self, state, episode):
        if episode % 3 == 0:
            return self.e_greedy_action(state,0.05)
        else:
            return self.e_greedy_action(state)

    def memorize(self, state, action, reward, terminal, train_step=True):
        """ It is a function that called when you want to add a transition to experience buffer
            Args:
              state: state before transition
              action: action made by agent
              reward: reward got by agent
              terminal: indicator whether next state is a terminal state
        """

        # Back up weights if time has come...
        if self.time_step % self.backup_steps == 0:
            # print 'Backing up networks...'
            directory = self.save_name + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.online_network.model.save_weights(directory + 'weights.h5', True)
            np.save(directory + 'step', [self.time_step])
            self.frozen_network.model.set_weights(np.copy(self.online_network.model.get_weights()))
            print 'Networks are backed up!'

        # Add transition to experience
        self.experience_state[self.buffer_last] = np.copy(state)
        self.experience_reward[self.buffer_last] = reward
        self.experience_action[self.buffer_last] = action
        self.experience_done[self.buffer_last] = terminal
        if train_step:
            # Set prior=0 to last transition because we don't want to optimize it
            if self.PRIORITIZED_XP_REPLAY:
                self.sum_prior -= self.experience_prob[self.buffer_last]
                self.experience_prob[self.buffer_last] = 0
            # Make a train_step if buffer is filled enough
            if self.buffer_size > min(self.buffer_max_size / 2, 50000) and self.time_step % self.train_every_steps == 0:
                self.train_step()

        # Reduce epsilon
        if self.n_observe < self.time_step < self.learning_time + self.n_observe:
            self.epsilon -= (1 - self.end_epsilon) / self.learning_time

        # Set max_prior to new transition after train_step
        if self.PRIORITIZED_XP_REPLAY:
            self.sum_prior += self.max_prior
            self.experience_prob[self.buffer_last] = self.max_prior
        # Add 1 to buffer_size, restrict it by buffer_max_size
        self.buffer_last = (self.buffer_last + 1) % self.buffer_max_size
        self.buffer_size = min(self.buffer_size+1,self.buffer_max_size)
        self.time_step += 1

    def train_step(self):

        if self.PRIORITIZED_XP_REPLAY:
            # Scale priority by sum of them
            probs = self.experience_prob / self.sum_prior
            # Choose transitions
            indexes_batch = np.random.choice(self.indexes, size=self.batch_size, p=probs, replace=False)
            # print indexes_batch
        else:
            # Sample transitions and avoid the last memorized transition
            indexes_batch = np.random.randint(0,self.buffer_size - 1, self.batch_size)
            indexes_batch = indexes_batch + (indexes_batch >= self.buffer_last).astype('int32')
        # Get batch of transitions
        state_batch = self.experience_state[indexes_batch]
        state_batch = np.copy(state_batch) / 255.
        action_batch = self.experience_action[indexes_batch]
        reward_batch = self.experience_reward[indexes_batch]
        # There is a simple trick to get "next_state" from next transition
        next_state_batch = self.experience_state[(indexes_batch + 1) % self.buffer_size]
        next_state_batch = np.copy(next_state_batch) / 255.
        done_batch = self.experience_done[indexes_batch]

        if self.PRIORITIZED_XP_REPLAY:
            prob_batch = self.experience_prob[indexes_batch]

        # Compute the q-value target
        output_frozen = self.frozen_network.get_output(next_state_batch)
        if self.DOUBLE_NETWORK:
            q_argmax_online = np.argmax(self.online_network.get_output(next_state_batch), axis=1)
            q_max_batch = output_frozen[self.batch_indexes, q_argmax_online]
        else:
            q_max_batch = np.max(output_frozen, axis=1)

        # Compute the target for network
        # If done=1, so next state is terminal and target q-value is equal to reward
        y_batch = (reward_batch + (1 - done_batch) * self.gamma * q_max_batch)
        #print y_batch
        # Reshape for QNeuralNetwork's functions
        y_batch = y_batch.reshape((self.batch_size, 1))
        action_batch = action_batch.reshape((self.batch_size, 1))
        state_batch = state_batch.reshape((self.batch_size,) + self.state_dim)

        if self.PRIORITIZED_XP_REPLAY:
            # Compute the weights for weighted update
            weights_batch = (self.buffer_size * prob_batch) ** -self.beta
            # Normalize weights by their's maximum
            weights_batch /= weights_batch.max()
            # Get info from Q-network and make train_step
            cost, error, Qs = self.online_network.train_step(y_batch, state_batch, action_batch, weights_batch)

            # Change the priorities of transitions we trained
            for i in range(self.batch_size):
                self.sum_prior = self.sum_prior + (abs(error[i]) + 0.0001) ** self.alpha - self.experience_prob[
                    indexes_batch[i]]
                self.experience_prob[indexes_batch[i]] = (abs(error[i]) + 0.0001) ** self.alpha
                # Try to refresh max_prior
            self.max_prior = np.max(self.experience_prob)
        else:
            cost, er,q,qs = self.online_network.train_step(y_batch, state_batch, action_batch)


        # Sometimes agent prints the cost value of batch
        if self.time_step % self.debug_steps == 0:
            print "Cost for the batch:" + str(cost), self.epsilon, np.mean(output_frozen), np.mean(
                output_frozen) - self.last_q
            self.last_q = np.mean(output_frozen)
