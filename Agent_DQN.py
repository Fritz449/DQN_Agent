import numpy as np
import QNeuralNetwork as NN
import tensorflow as tf
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
        self.end_epsilon = 0.01
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
        self.sess = tf.InteractiveSession()
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
        self.online_network = NN.QNeuralNetwork('Online-net', state_dim, action_dim, batch_size=batch_size,
                                                learning_rate=learning_rate, DUELING_ARCHITECTURE=DUELING_ARCHITECTURE)
        print 'Frozen-network initializing...'
        self.frozen_network = NN.QNeuralNetwork('Frozen-net', state_dim, action_dim, batch_size=batch_size,
                                                learning_rate=learning_rate, DUELING_ARCHITECTURE=DUELING_ARCHITECTURE)

        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver(tf.all_variables())
        # Try to load weights if we made an agent for our task before
        try:
            print "Try to load networks from files..."
            self.load_agent()
        except:
            print "Training a new model"

        print 'Networks initialized.'

        # Initializing of experience buffer
        self.experience_state = np.zeros((self.buffer_max_size,) + self.state_dim, dtype='float32') * 1.0
        self.experience_action = np.zeros(self.buffer_max_size) * 1.0
        self.experience_reward = np.zeros(self.buffer_max_size) * 1.0
        self.experience_done = np.zeros(self.buffer_max_size) * 1.0
        if self.PRIORITIZED_XP_REPLAY:
            self.experience_prob = np.zeros(self.buffer_max_size) * 1.0

    def greedy_action(self, state):
        # This is a function for environment-agent interaction
        act = np.argmax(self.online_network.get_output(state)[:self.action_dim])
        # print self.online_network.get_output(state)[:self.action_dim]
        return act

    def e_greedy_action(self, state):
        # This is a function for environment-agent interaction
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            return self.greedy_action(state)

    def action(self, state, episode):
        if episode % 100 == 0:
            return self.e_greedy_action(state)
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
            self.update_target_network()
            print 'Networks are backed up!'

        # Add transition to experience
        self.experience_state[self.time_step % self.buffer_max_size] = np.copy(state)
        self.experience_reward[self.time_step % self.buffer_max_size] = reward
        self.experience_action[self.time_step % self.buffer_max_size] = action
        self.experience_done[self.time_step % self.buffer_max_size] = terminal

        if train_step:
            # Set prior=0 to last transition because we don't want to optimize it
            if self.PRIORITIZED_XP_REPLAY:
                self.sum_prior -= self.experience_prob[self.time_step % self.buffer_max_size]
                self.experience_prob[self.time_step % self.buffer_max_size] = 0
            # Make a train_step if buffer is filled enough
            if self.time_step > self.buffer_max_size / 2 and self.time_step % self.train_every_steps == 0:
                self.train_step()

        # Reduce epsilon
        if self.n_observe < self.time_step < self.learning_time + self.n_observe:
            self.epsilon -= (1 - 0.01) / self.learning_time

        # Set max_prior to new transition after train_step
        if self.PRIORITIZED_XP_REPLAY:
            self.sum_prior += self.max_prior
            self.experience_prob[self.time_step % self.buffer_max_size] = self.max_prior
        # Add 1 to buffer_size, restrict it by buffer_max_size
        self.buffer_size += int(self.buffer_size < self.buffer_max_size)
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
            indexes_batch = np.random.choice(range(self.buffer_size - 1), self.batch_size, replace=False)
            indexes_batch = indexes_batch + (indexes_batch >= self.time_step % self.buffer_max_size).astype('int32')

        # Get batch of transitions
        state_batch = self.experience_state[indexes_batch]
        action_batch = self.experience_action[indexes_batch]
        reward_batch = self.experience_reward[indexes_batch]
        # There is a simple trick to get "next_state" from next transition
        next_state_batch = self.experience_state[(indexes_batch + 1) % self.buffer_size]
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
        if self.PRIORITIZED_XP_REPLAY:
            # Compute the weights for weighted update
            weights_batch = (self.buffer_size * prob_batch) ** -self.beta
            # Normalize weights by their's maximum
            weights_batch /= weights_batch.max()
            # Get info from Q-network and make train_step
            cost, error = self.online_network.train_step(y_batch, state_batch, action_batch, weights_batch)

            # Change the priorities of transitions we trained
            for i in range(self.batch_size):
                self.sum_prior = self.sum_prior + (abs(error[i]) + 0.0001) ** self.alpha - self.experience_prob[
                    indexes_batch[i]]
                self.experience_prob[indexes_batch[i]] = (abs(error[i]) + 0.0001) ** self.alpha
                # Try to refresh max_prior
            self.max_prior = np.max(self.experience_prob)
        else:
            cost, error = self.online_network.train_step(y_batch, state_batch, action_batch)

        # Sometimes agent prints the cost value of batch
        if self.time_step % self.debug_steps == 0:
            print "Cost for the batch:" + str(cost), self.epsilon, np.mean(output_frozen)

    def save_agent(self, epoch, name='DQN_w'):
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        self.saver.save(self.sess, name + '/model.ckpt',
                        global_step=epoch + 1)

    def load_agent(self, name='DQN_w'):
        ckpt = tf.train.get_checkpoint_state(name + '/')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print 'found a checkpoint'
        else:
            print 'no checkpoints founded'

    def update_target_network(self):
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith('Online-net')]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith('Frozen-net')]
        e2_params = sorted(e2_params, key=lambda v: v.name)
        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        self.save_agent()
        self.sess.run(update_ops)
