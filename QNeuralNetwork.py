import numpy as np
from keras import backend as Theano
from keras.layers import Dense, Input, Convolution2D, Flatten
from keras.models import Model
from keras.optimizers import RMSprop


class QNeuralNetwork:
    # def create_model(self):
    #     self.state_in = Input(self.state_dim)
    #     self.l1 = Convolution2D(16, 8, 8, activation='relu', subsample=(4, 4), border_mode='same')(self.state_in)
    #     self.l2 = Convolution2D(32, 4, 4, activation='relu', subsample=(2, 2), border_mode='same')(self.l1)
    #     self.h = Flatten()(self.l2)
    #     self.h = Dense(256, activation='relu')(self.h)
    #     self.q = Dense(self.action_dim)(self.h)
    #     self.model = Model(self.state_in, self.q)

    def create_model(self):
        # This is the place where neural network model initialized
        self.state_in = Input(self.state_dim)  # This layer is required for any network.
        self.h = Dense(30, activation='relu')(self.state_in)  # Hidden layer of network.
        self.q = Dense(self.action_dim)(self.h)  # Output layer of network
        self.model = Model(self.state_in, self.q)  # Complete the model

    def __init__(self, state_dim, action_dim, batch_size=32, learning_rate=0.0001):
        """ Initialize Q-network.
            Args:
              state_dim: dimensionality of space of states
              action_dim: dimensionality of space of actions
              batch_size: size of minibatch for network's train
              learning_rate: learning rate of optimizer
        """

        # Assign network features
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # Create input for training
        self.actions = Input(shape=(self.batch_size,), dtype='int32')
        self.target = Input(shape=(self.batch_size,), dtype='int32')
        self.weights = Input(shape=(self.batch_size,), dtype='float32')  # These weights are for weighted update

        # Initialize model
        self.create_model()

        # Compute q-values
        self.Qs = self.model(self.state_in)
        # Make a function for get output of network
        self.q_value = Theano.function([self.state_in], self.Qs)
        # Get q-values for corresponding actions
        self.q_output = self.Qs[np.array(range(self.batch_size)), self.actions.reshape((self.batch_size,))]

        # Compute TD-error
        self.error = (self.q_output - self.target.reshape((self.batch_size,)))
        # Make a MSE-cost function
        self.cost = (self.weights * (self.error ** 2)).mean()
        # Initialize an optimizer
        self.opt = RMSprop(self.learning_rate)
        self.params = self.model.trainable_weights
        self.updates = self.opt.get_updates(self.params, [], self.cost)

        # Make a function to update weights and get information about cost an TD-errors
        self.tr_step = Theano.function([self.target, self.state_in, self.actions, self.weights],  # Input
                                       [self.cost, self.error],  # Output when make a training step
                                       updates=self.updates)  # Update weights

    def get_output(self, state):
        # This is a function for simple agent-network interaction

        return self.q_value([state])

    def q_actions(self, state, actions):
        # This is a function for simple agent-network interaction

        return self.q_value([state])[np.array(range(self.batch_size)), actions]

    def train_step(self, target, state_in, actions, weights=None):
        """ This is a function which agent calls when want to train network.
        If there is no prioritized xp-replay there is no weighted update and weights are set as 1
        """
        if weights is None:
            weights = np.ones((1, state_in.shape[0]))
        weights = weights.reshape(1, state_in.shape[0])
        return self.tr_step([target, state_in, actions, weights])
