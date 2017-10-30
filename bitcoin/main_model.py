import os
import numpy as np
import tensorflow as tf
import time
import re
import matplotlib.pyplot as plt
import math



class CNN_Model:
    def __init__(self, sess, model_name):
        self.sess = sess
        self.model_name = model_name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.model_name):
            with tf.name_scope('input_layer'):
                self.learning_rate = 0.001
                self.training = tf.placeholder(tf.bool, name='training')
                self.regularizer = tf.contrib.layers.l2_regularizer(0.0005)

                self.X = tf.placeholder(dtype=tf.float32, shape=[None, 100])
                X_data = tf.reshape(self.X, [-1, 1, 100, 1])
                self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 100])

            with tf.name_scope('conv_layer'):
                self.W1_conv = tf.get_variable(name='W1_conv', shape=[1, 20, 1, 100], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L1_conv = tf.nn.conv2d(input=X_data, filter=self.W1_conv, strides=[1, 1, 1, 1], padding='VALID')  # 1x100 -> 1x81
                self.L1_conv = self.BN(input=self.L1_conv, training=self.training, name='L1_conv_BN')
                self.L1_conv = self.parametric_relu(self.L1_conv, 'R1_conv')

                self.W2_conv = tf.get_variable(name='W2_conv', shape=[1, 20, 100, 200], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L2_conv = tf.nn.conv2d(input=self.L1_conv, filter=self.W2_conv, strides=[1, 1, 1, 1], padding='VALID')  # 1x81 -> 1x62
                self.L2_conv = self.BN(input=self.L2_conv, training=self.training, name='L2_conv_BN')
                self.L2_conv = self.parametric_relu(self.L2_conv, 'R2_conv')

                self.W3_conv = tf.get_variable(name='W3_conv', shape=[1, 20, 200, 300], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L3_conv = tf.nn.conv2d(input=self.L2_conv, filter=self.W3_conv, strides=[1, 1, 1, 1], padding='VALID')  # 1x62 -> 1x43
                self.L3_conv = self.BN(input=self.L3_conv, training=self.training, name='L3_conv_BN')
                self.L3_conv = self.parametric_relu(self.L3_conv, 'R3_conv')

                self.W4_conv = tf.get_variable(name='W4_conv', shape=[1, 20, 300, 400], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L4_conv = tf.nn.conv2d(input=self.L3_conv, filter=self.W4_conv, strides=[1, 1, 1, 1], padding='VALID')  # 1x43 -> 1x24
                self.L4_conv = self.BN(input=self.L4_conv, training=self.training, name='L4_conv_BN')
                self.L4_conv = self.parametric_relu(self.L4_conv, 'R4_conv')
                self.L4_conv = tf.reshape(self.L4_conv, [-1, 1 * 24 * 400])

            with tf.name_scope('fc_layer'):
                self.W1_fc = tf.get_variable(name='W1_fc', shape=[1 * 24 * 400, 1000], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b1_fc = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b1_fc'))
                self.L1_fc = tf.matmul(self.L4_conv, self.W1_fc) + self.b1_fc
                self.L1_fc = self.BN(input=self.L1_fc, training=self.training, name='L1_fc_BN')
                self.L1_fc = self.parametric_relu(self.L1_fc, 'R1_fc')

                self.W2_fc = tf.get_variable(name='W2_fc', shape=[1000, 1000], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b2_fc = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b2_fc'))
                self.L2_fc = tf.matmul(self.L1_fc, self.W2_fc) + self.b2_fc
                self.L2_fc = self.BN(input=self.L2_fc, training=self.training, name='L2_fc_BN')
                self.L2_fc = self.parametric_relu(self.L2_fc, 'R2_fc')

            self.W_out = tf.get_variable(name='W_out', shape=[1000, 100], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[100], name='b_out'))
            self.logits = tf.matmul(self.L2_fc, self.W_out) + self.b_out

            self.reg_cost = tf.reduce_sum([self.regularizer(train_var) for train_var in tf.get_variable_scope().trainable_variables() if re.search(self.model_name+'\/W', train_var.name) is not None])
            self.cost = self.huber_loss(0.5 * tf.reduce_sum(tf.square(self.logits - self.Y)) + 0.0005 * self.reg_cost)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

    def huber_loss(self, loss):
        return tf.where(tf.abs(loss) <= 1.0, 0.5 * tf.square(loss), tf.abs(loss) - 0.5)

    def BN(self, input, training, name):
        return tf.contrib.layers.batch_norm(input, decay=0.99, scale=True, is_training=training, updates_collections=None, scope=name)

    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})




cnn_input_size = 10 * 10
batch_size = 100
epochs = 20
step_size = batch_size * cnn_input_size

