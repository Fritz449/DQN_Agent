import numpy as np
import tensorflow as tf
import os

def weight_variable(name, shape, stddev=1. / 256 / 256):
    initial = tf.random_normal(shape, stddev=stddev)
    return tf.get_variable(name, initializer=initial, dtype='float32')


def bias_variable(name, shape):
    initial = tf.constant(0.00001, shape=shape)
    return tf.get_variable(name, initializer=initial, dtype='float32')


def batch_norm(scope, x, n_out, phase_train):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def conv_layer(name, input_tensor, kernel_shape, train, relu=True, alpha=0, max_pooling=None, b_norm=True,
               reflect=False):
    with tf.variable_scope(name):
        kernel = weight_variable("kernel", kernel_shape, 1. / kernel_shape[1] / kernel_shape[2] / kernel_shape[0])
        biases = bias_variable("bias", (kernel_shape[3],))
        if reflect:
            input_tensor = tf.pad(input_tensor, [[0, 0], [kernel_shape[1] / 2, kernel_shape[1] / 2],
                                                 [kernel_shape[1] / 2, kernel_shape[1] / 2], [0, 0]], mode='REFLECT')
            convolution = tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='VALID')
        else:
            convolution = tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME')
        end = convolution + biases
        if b_norm:
            end = batch_norm(name + 'bn', end, kernel_shape[3], train)
        if relu:
            end = tf.maximum(tf.mul(end, alpha), end)

        if not (max_pooling is None):
            return tf.nn.max_pool(end, ksize=[1, max_pooling, max_pooling, 1], strides=[1, max_pooling, max_pooling, 1],
                                  padding='SAME')
        return end


def fc_layer(name, input_tensor, n_out, relu=True, alpha=0):
    with tf.variable_scope(name):
        print input_tensor.get_shape()
        weights = weight_variable("weights", shape=(int(input_tensor.get_shape()[1]), n_out),
                                  stddev=1. / int(input_tensor.get_shape()[1]) / n_out)
        biases = bias_variable("bias", shape=(n_out,))
        out = tf.matmul(input_tensor, weights) + biases

        if relu:
            out = tf.maximum(tf.mul(out, alpha), out)
        return out


class QNeuralNetwork:
    def create_conv_model(self):
        # This is the place where neural network model initialized
        self.state_in = tf.placeholder(shape=((None,) + self.state_dim), dtype='float32')
        print self.state_in.get_shape()
        self.l1 = conv_layer(self.name + 'conv1', self.state_in, (8, 8, 4, 32), max_pooling=4, train=self.train)
        self.l2 = conv_layer(self.name + 'conv2', self.l1, (4, 4, 32, 64), max_pooling=2, train=self.train)
        self.l3 = conv_layer(self.name + 'conv3', self.l2, (3, 3, 64, 64), max_pooling=1, train=self.train)
        print self.l3.get_shape()
        self.h = tf.reshape(self.l3, [tf.shape(self.l3)[0],
                                      int(self.l3.get_shape()[1] * self.l3.get_shape()[2] * self.l3.get_shape()[3])])
        print self.h.get_shape()
        if self.DUELING_ARCHITECTURE:
            self.hida = fc_layer(self.name + 'hida', self.h, 256)
            self.hidv = fc_layer(self.name + 'hidv', self.h, 256)
            self.v = fc_layer(self.name + 'value', self.hidv, 1)
            self.a = fc_layer(self.name + 'advan', self.hida, self.action_dim)
            self.q = self.v + self.a - tf.reduce_mean(self.a)
        else:
            self.hid = fc_layer(self.name + 'hid', self.h, 256)
            self.q = fc_layer(self.name + 'q-value', self.hid, self.action_dim)
        return self.q

    def create_fc_model(self):
        # This is the place where neural network model initialized
        self.state_in = tf.placeholder(shape=((None,) + self.state_dim), dtype='float32')
        if self.DUELING_ARCHITECTURE:
            self.hida = fc_layer('hida', self.state_in, 256)
            self.hidv = fc_layer('hidv', self.state_in, 256)
            self.v = fc_layer('value', self.hidv, 1)
            self.a = fc_layer('advan', self.hida, self.action_dim)
            self.q = self.v + self.a - tf.reduce_mean(self.a)
        else:
            self.hid = fc_layer('hid', self.state_in, 256)
            self.q = fc_layer('q-value', self.hid, self.action_dim)
        return self.q

    def __init__(self, name, state_dim, action_dim, batch_size=32, learning_rate=0.1, DUELING_ARCHITECTURE=False):
        """ Initialize Q-network.
            Args:
              state_dim: dimensionality of space of states
              action_dim: dimensionality of space of actions
              batch_size: size of minibatch for network's train
              learning_rate: learning rate of optimizer
              DUELING_ARCHITECTURE: dueling network architecture activation
              conv_model: whether agent uses convolutional neural network
        """

        # Assign network features
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.DUELING_ARCHITECTURE = DUELING_ARCHITECTURE
        # Create input for training
        self.actions = tf.placeholder(shape=(self.batch_size,), dtype='int32')
        self.target = tf.placeholder(shape=(self.batch_size,), dtype='float32')
        # These weights are for weighted update
        self.weights = tf.placeholder(shape=(self.batch_size,),
                                      dtype='float32')
        # Train/Using phase
        self.train = tf.placeholder(dtype='bool')

        # Initialize model and compute q-values
        if len(state_dim) == 3:
            self.Qs = self.create_conv_model()
        else:
            self.Qs = self.create_fc_model()

        # Get q-values for corresponding actions
        idx_flattened = tf.range(0, self.batch_size) * self.action_dim + self.actions
        self.q_output = tf.gather(tf.reshape(self.Qs, [-1]), idx_flattened)
        # Compute TD-error
        self.error = (self.q_output - self.target)
        # Make a MSE-cost function
        self.loss = tf.reduce_mean(self.weights * (self.error ** 2))
        # Initialize an optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def get_output(self, state):
        # This is a function for simple agent-network interaction
        return self.Qs.eval(feed_dict={self.state_in: state, self.train: False})

    def q_actions(self, state, actions):
        # This is a function for simple agent-network interaction
        return self.q_output.eval(feed_dict={self.state_in: state, self.actions: actions, self.train: False})

    def train_step(self, target, state_in, actions, weights=None):
        """ This is a function which agent calls when want to train network.
        If there is no prioritized xp-replay there is no weighted update and weights are set as 1
        """
        if weights is None:
            weights = np.ones(state_in.shape[0], )
        weights = weights.reshape(state_in.shape[0],)
        return self.sess.run([self.optimizer, self.loss, self.error], feed_dict={self.actions: actions,
                                                                                 self.state_in: state_in,
                                                                                 self.weights: weights,
                                                                                 self.target: target,
                                                                                 self.train: True
                                                                                 })[1:]

    # def save_net(self, epoch):
    #     if not os.path.exists(self.name + '/'):
    #         os.makedirs(self.name + '/')
    #     self.saver.save(self.sess, self.name + '/model.ckpt',
    #                     global_step=epoch + 1)
    #
    # def load_net(self, name='Online-net'):
    #     ckpt = tf.train.get_checkpoint_state(name + '/')
    #     if ckpt and ckpt.model_checkpoint_path:
    #         self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    #         print 'found a checkpoint'
    #     else:
    #         print 'no checkpoints founded'


