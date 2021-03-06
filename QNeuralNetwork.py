import numpy as np
from keras import backend as Theano
from keras.layers import Dense, Input, Convolution2D, Flatten, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adadelta, RMSprop, Adam, SGD
from keras.regularizers import l1, l2
from keras.initializations import normal, glorot_uniform

# def my_init(shape, name=None):
#     return normal(shape, scale=0.01, name=name)

class QNeuralNetwork:
    def create_model(self):
        my_init = 'glorot_uniform'
        # This is the place where neural network model initialized
        self.state_in = Input(self.state_dim)
        # self.state_inp = BatchNormalization()(self.state_in)
        self.l1 = Convolution2D(32, 8, 8, activation='relu', init=my_init, subsample=(4, 4), border_mode='same')(
            self.state_in)
        # self.l1bn = BatchNormalization()(self.l1)
        self.l2 = Convolution2D(64, 4, 4, activation='relu', init=my_init, subsample=(2, 2), border_mode='same')(
            self.l1)
        # self.l2bn = BatchNormalization()(self.l2)
        self.l3 = Convolution2D(64, 3, 3, activation='relu', init=my_init, subsample=(1, 1), border_mode='same')(
            self.l2)
        # self.l3bn = BatchNormalization()(self.l3)
        self.h = Flatten()(self.l3)

        if self.DUELING_ARCHITECTURE:
            self.hida = Dense(256, init=my_init, activation='relu')(self.h)
            self.hidv = Dense(256, init=my_init, activation='relu')(self.h)
            self.v = Dense(1)(self.hidv)
            self.a = Dense(self.action_dim)(self.hida)
            self.q = merge([self.a, self.v], mode='concat')
        else:
            self.hid = Dense(512, init=my_init, activation='relu')(self.h)
            self.q = Dense(self.action_dim, init=my_init)(self.hid)
        self.model = Model(self.state_in, self.q)

    # def create_model(self):
    #     # This is the place where neural network model initialized
    #     self.state_in = Input(self.state_dim)  # This layer is required for any network.
    #     if self.DUELING_ARCHITECTURE:
    #         self.hida = Dense(10, activation='relu')(self.state_in)
    #         self.hidv = Dense(10, activation='relu')(self.state_in)
    #         self.v = Dense(1)(self.hidv)
    #         self.a = Dense(self.action_dim)(self.hida)
    #         self.q = merge([self.a, self.v], mode='concat')
    #     else:
    #         self.hid = Dense(64, activation='relu', init='lecun_uniform')(self.state_in)
    #         self.q = Dense(self.action_dim, init='lecun_uniform')(self.hid)
    #     self.model = Model(self.state_in, self.q)  # Complete the model

    def __init__(self, state_dim, action_dim, batch_size=32, learning_rate=1., DUELING_ARCHITECTURE=False):
        """ Initialize Q-network.
            Args:
              state_dim: dimensionality of space of states
              action_dim: dimensionality of space of actions
              batch_size: size of minibatch for network's train
              learning_rate: learning rate of optimizer
              DUELING_ARCHITECTURE: dueling network architecture activation
        """

        # Assign network features
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.DUELING_ARCHITECTURE = DUELING_ARCHITECTURE
        # Create input for training
        self.actions = Input(shape=(self.batch_size,), dtype='int32')
        self.target = Input(shape=(self.batch_size,), dtype='float32')
        self.weights = Input(shape=(self.batch_size,), dtype='float32')  # These weights are for weighted update

        # Initialize model
        self.create_model()

        # Compute q-values

        self.Qs = self.model.layers[-1].output

        # Make a function for get output of network
        self.q_value = Theano.function([self.state_in], self.Qs[:, :self.action_dim])
        # Get q-values for corresponding actions
        tmp = np.arange(self.batch_size)

        actionmask = Theano.T.eq(Theano.T.arange(self.action_dim).reshape((1, -1)),
                               self.actions.reshape((-1, 1))).astype(Theano.T.config.floatX)

        if DUELING_ARCHITECTURE:
            self.a_output = self.Qs[tmp, self.actions.reshape((self.batch_size,))]
            self.v_output = self.Qs[tmp, -1]
            self.q_output = self.a_output + self.v_output - self.Qs[tmp,
                                                            : self.action_dim].mean(axis=1)

        else:
            self.q_output = Theano.T.sum(self.Qs * actionmask, axis=1)

        # Compute TD-error
        self.error = (self.q_output - self.target.reshape((self.batch_size,)))
        # Make a MSE-cost function
        self.error_ = self.weights * self.error

        quadratic_part = Theano.T.minimum(abs(self.error_), 1.)
        linear_part = abs(self.error_) - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + 1. * linear_part

        self.cost = Theano.T.sum(loss)/self.batch_size
        # Initialize an optimizer
        self.opt = RMSprop(lr=self.learning_rate, rho=0.95, epsilon=1e-6)
        self.params = self.model.trainable_weights
        self.updates = self.opt.get_updates(self.params, [], self.cost)
        # Make a function to update weights and get information about cost an TD-errors
        self.tr_step = Theano.function(
            [self.target, self.state_in, self.actions, self.weights],  # Input
            [self.cost, self.error, self.q_output, self.Qs,self.error_],  # Output when make a training step
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
        return self.tr_step([target, state_in, actions, weights])
