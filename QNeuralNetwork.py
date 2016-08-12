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
        self.state_in = Input(self.state_dim)
        self.h = Dense(30, activation='relu')(self.state_in)
        self.q = Dense(self.action_dim)(self.h)
        self.model = Model(self.state_in, self.q)

    def __init__(self, state_dim, action_dim, batch_size=32, name='Network'):
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size

        self.actions = Input(shape=(self.batch_size,), dtype='int32')
        self.target = Input(shape=(self.batch_size,), dtype='int32')
        self.weights = Input(shape=(self.batch_size,), dtype='float32')

        self.create_model()

        self.QS = self.model(self.state_in)
        self.q_output = self.QS[np.array(range(self.batch_size)), self.actions.reshape((self.batch_size,))]

        self.error = (self.q_output - self.target.reshape((self.batch_size,)))
        self.cost = (self.weights * (self.error ** 2)).mean()
        self.opt = RMSprop(0.0001)
        self.params = self.model.trainable_weights
        self.updates = self.opt.get_updates(self.params, [], self.cost)

        self.tr_step = Theano.function([self.target, self.state_in, self.actions, self.weights],
                                       [self.cost, self.error],  # debug output
                                       updates=self.updates)  # update weights

        self.q_value = Theano.function([self.state_in], self.QS)  # get output of network

    def get_output(self, state):
        return self.q_value([state])

    def q_actions(self, state, actions):
        return self.q_value([state])[np.array(range(self.batch_size)), actions]

    def train_step(self, target, state_in, actions, weights=None):
        if weights is None:
            weights = np.ones((1, state_in.shape[0]))
        weights = weights.reshape(1, state_in.shape[0])
        return self.tr_step([target, state_in, actions, weights])
