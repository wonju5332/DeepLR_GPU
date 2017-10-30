import tensorflow as tf
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
from math import sqrt
import numpy as np

'''
learning rate = 1-4
epoch = 200
optimizer = Adam
weight_init = mu 0 var 1
bias_init = 0

"Reg"
l2 alpha = 1e-5
alpha-dropout

activation = SELU



Dense_BC + SNN ( SeLU + alpha_dropout )



SNN :
	1. scale inputs to zero mean and unit variance
	2. use selus
	3. initialize weights with stddev sqrt(1/n)
    4. use alpha dropout

'''



class Model():
    def __init__(self, sess, depth):
        self.sess = sess
        self.N = int((depth - 4) / 3)      #depth = 40  -> 37/3 -> 12.333 --> 12
        self.growthRate = 12
        self.compression_factor = 0.5
        self._build_graph()

    def _build_graph(self):
        with tf.name_scope('initialize_scope'):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 1024], name='X_data')
            reshape_x = tf.reshape(self.X, shape=[-1, 32, 32, 1])  #gray이기때문에 32x32x1
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_data')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.dropout_rate = tf.placeholder(dtype=tf.float32, name='dropout_rate')
            self.learning_rate = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)  # 논문반영 lr = 1e-4


        def conv(l, kernel, channel, stride):
            return layers.conv2d(inputs=l, num_outputs=channel, kernel_size=kernel, stride=stride, padding='SAME',
                          weights_initializer=layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'), weights_regularizer=layers.l2_regularizer(1e-5))


        def add_layer(name, l):
            with tf.variable_scope(name):
                '''bottleneck layer (DenseNet-B)'''
                c = self.selu(l, 'bottleneck')
                c = conv(c, 1, 4 * self.growthRate, 1)  # 4k, output
                c = self.dropout_selu(x=c, rate=self.dropout_rate, training=self.training)
                '''basic dense layer (Factorization)'''
                c = conv(c, 3, self.growthRate, 1)  # k, output
                c = self.dropout_selu(x=c, rate=self.dropout_rate, training=self.training)
                l = tf.concat([c, l], axis=3)
            return l


        def add_transition(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name):
                '''compression transition layer (DenseNet-C)'''
                # l = layers.batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                l = self.selu(l,'selu_'+str(name))
                # l = tf.nn.relu(l, 'transition')
                l = conv(l, 3, int(in_channel * self.compression_factor), 1)
                l = layers.avg_pool2d(inputs=l, kernel_size=[2, 2], stride=2, padding='SAME')
                l = self.dropout_selu(x=l, rate=self.dropout_rate, training=self.training)
                # l = layers.dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)  #todo  alpha로
            return l

        def dense_net():
            l = conv(reshape_x, 3, 16, 1)  #todo shape = 32x32
            #  -> 32x32x16
            # l = avg_pool2d(inputs=l, kernel_size=[2, 2], stride=2, padding='SAME')

            with tf.variable_scope('dense_block1'):
                for i in range(self.N):   #12번
                    l = add_layer('dense_layer_{}'.format(i), l)
                l = add_transition('transition1', l)

            with tf.variable_scope('dense_block2'):
                for i in range(self.N):
                    l = add_layer('dense_layer_{}'.format(i), l)
                l = add_transition('transition2', l)

            with tf.variable_scope('dense_block3'):
                for i in range(self.N):
                    l = add_layer('dense_layer_{}'.format(i), l)

            # l = layers.batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
            # l = self.parametric_relu(l, 'output')
            # l = tf.nn.elu(l, 'output')
            # l = tf.nn.relu(l, 'output')
            l = self.selu(l, name='selu_pre_global')
            l = layers.avg_pool2d(inputs=l, kernel_size=[8, 8], stride=1, padding='VALID')
            l = tf.reshape(l, shape=[-1, 1 * 1 * 256])
            l = self.dropout_selu(x=l, rate=self.dropout_rate, training=self.training)
            # l = layers.dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
            logits = layers.fully_connected(inputs=l, num_outputs=10, activation_fn=None,
                                     weights_initializer=layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'),
                                            weights_regularizer=layers.l2_regularizer(1e-5))

            return logits

        self.logits = dense_net()
        self.prob = tf.nn.softmax(logits=self.logits, name='output')
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        loss = tf.reduce_mean(loss, name='cross_entropy_loss')
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([loss] + reg_losses, name='loss')
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.y, 1)), dtype=tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False, self.dropout_rate: 0.0})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.y: y_test, self.training: False, self.dropout_rate: 0.0})

    def train(self, x_data, y_data):
        return self.sess.run([self.accuracy, self.optimizer],
                             feed_dict={self.X: x_data, self.y: y_data, self.training: True, self.dropout_rate: 0.05})   #todo alpha dropout rate = 0.05 (논문반영)

    def validation(self, x_test, y_test):
        return self.sess.run([self.loss, self.accuracy], feed_dict={self.X: x_test, self.y: y_test, self.training: False, self.dropout_rate: 0.0})

    # def snn_init(self,kernel):
    #     # Determine number of input features from shape
    #     f_in = kernel[0]**2 if len(kernel) == 2 else kernel[0]
    #     sdev = sqrt(1 / f_in)
    #     init = tf.truncated_normal_initializer(stddev=sdev, dtype=tf.float32)
    #     return init


    def selu(self, x, name="selu"):
        """ When using SELUs you have to keep the following in mind:
        # (1) scale inputs to zero mean and unit variance
        # (2) use SELUs
        # (3) initialize weights with stddev sqrt(1/n)
        # (4) use SELU dropout
        """
        with ops.name_scope(name) as scope:
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

    def dropout_selu(self, x, rate, alpha=-1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                     noise_shape=None, seed=None, name=None, training=False):

        """Dropout to a value with rescaling."""
        def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
            keep_prob = 1.0 - rate
            x = ops.convert_to_tensor(x, name="x")
            if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
                raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                 "range (0, 1], got %g" % keep_prob)

            keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
            keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
            alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
            alpha.get_shape().assert_is_compatible_with(tensor_shape.scalar())

            if tensor_util.constant_value(keep_prob) == 1:
                return x
            noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
            random_tensor = keep_prob
            random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
            binary_tensor = math_ops.floor(random_tensor)
            ret = x * binary_tensor + alpha * (1 - binary_tensor)
            a = math_ops.sqrt(fixedPointVar / (
            keep_prob * ((1 - keep_prob) * math_ops.pow(alpha - fixedPointMean, 2) + fixedPointVar)))
            b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
            ret = a * ret + b
            ret.set_shape(x.get_shape())
            return ret

        with ops.name_scope(name, "dropout", [x]) as name:
            return utils.smart_cond(training,
                                    lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
                                    lambda: array_ops.identity(x))



