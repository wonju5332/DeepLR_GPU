import tensorflow as tf
import numpy as np
import time

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()
        self.early_stop_count = 0
        self.epoch = 0

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('input_layer') as scope:
                ########################################################################################################
                ## �� Dropout
                ##  - �������� ��带 �����Ͽ� �Է°� ��� ������ ������ �����ϴ� ���.
                ##  - ���� �����Ϳ� �������� �Ǵ� ���� �����ִ� ����.
                ##  �� �н� : 0.5, �׽�Ʈ : 1
                ########################################################################################################
                self.dropout_rate = tf.Variable(tf.constant(value=0.5), name='dropout_rate')
                self.training = tf.placeholder(tf.bool, name='training')

                self.X = tf.placeholder(tf.float32, [None, 50, 50, 3], name='x_data')
                self.Y = tf.placeholder(tf.float32, [None, 2], name='y_data')

            ############################################################################################################
            ## �� Convolution ���� - 1
            ##  �� �ռ��� ���� �� filter: (7, 7), padding: VALID output: 20 ��, �ʱⰪ: He
            ##  �� Ȱ��ȭ �Լ� �� Parametric Relu
            ##  �� Ǯ�� ����   �� Max Pooling
            ##  �� ��� �ƿ� ����
            ############################################################################################################
            with tf.name_scope('conv_sub_1-1') as scope:
                self.W1_con_sub = tf.get_variable(name='W1_con_sub', shape = [3, 3, 3, 20], dtype=tf.float32,
                                                  initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L1_con_sub = tf.nn.conv2d(name='L1_con_sub', input=self.X, filter=self.W1_con_sub,
                                               strides=[1,1,1,1], padding='VALID')  # 50X50 -> 46X46
                self.L1_BN_sub = self.BN(input=self.L1_con_sub, scale=True, training=self.training, name='L1_BN_sub')
                self.L1_con_sub = self.parametric_relu(_x=self.L1_BN_sub, name='RU1_con_sub')

            with tf.name_scope('conv_sub_1-2') as scope:
                self.W2_con_sub = tf.get_variable(name='W2_con_sub', shape = [3, 3, 20, 60], dtype=tf.float32,
                                                  initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L2_con_sub = tf.nn.conv2d(name='L2_con_sub', input=self.L1_con_sub, filter=self.W2_con_sub,
                                               strides=[1,1,1,1], padding='VALID')  # 46X46 -> 42X42
                self.L2_BN_sub = self.BN(input=self.L2_con_sub, scale=True, training=self.training, name='L2_BN_sub')
                self.L2_con_sub = self.parametric_relu(_x=self.L2_BN_sub, name='Ru2_con_sub')
                self.L2_con_sub = tf.nn.max_pool(value=self.L2_con_sub, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')  # 42X42 -> 21X21

            with tf.name_scope('conv_layer2') as scope:
                self.W3_con_layer = tf.get_variable(name='W3_con_layer', shape= [3, 3, 60, 120], dtype=tf.float32,
                                                    initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L3_con_layer = tf.nn.conv2d(name='L3_con_layer', input=self.L2_con_sub, filter=self.W3_con_layer,
                                                 strides=[1, 1, 1, 1], padding='VALID')  # 21X21 -> 19x19
                self.L3_BN_sub = self.BN(input=self.L3_con_layer, scale=True, training=self.training, name='L3_BN_sub')
                self.L3_con_layer = self.parametric_relu(_x=self.L3_BN_sub, name='Ru3_con_sub')
                self.L3_con_layer = tf.nn.max_pool(value=self.L3_con_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 19X19 -> 10X10

            with tf.name_scope('conv_layer3') as scope:
                self.W4_con_layer = tf.get_variable(name='W4_con_layer', shape= [3, 3, 120, 220], dtype=tf.float32,
                                                    initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L4_con_layer = tf.nn.conv2d(name='L4_con_layer', input=self.L3_con_layer, filter=self.W4_con_layer,
                                                 strides=[1, 1, 1, 1], padding='VALID')  # 10X10 -> 8x8
                self.L4_BN_sub = self.BN(input=self.L4_con_layer, scale=True, training=self.training, name='L4_BN_sub')
                self.L4_con_layer = self.parametric_relu(_x=self.L4_BN_sub, name='Ru4_con_sub')
                self.L4_con_layer = tf.nn.max_pool(value=self.L4_con_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')  # 8X8 -> 4X4
                self.L4_con_layer = tf.reshape(self.L4_con_layer, [-1, 4*4*220])
            ############################################################################################################
            ## �� fully connected ���� - 1
            ##  �� ����ġ      �� shape: (4 * 4 * 320, 625), output: 625 ��, �ʱⰪ: He
            ##  �� ����        �� shape: 625, �ʱⰪ: 0.001
            ##  �� Ȱ��ȭ �Լ� �� Parametric Relu
            ##  �� ��� �ƿ� ����
            ############################################################################################################
            with tf.name_scope('fc_layer1') as scope:
                self.W_fc1 = tf.get_variable(name='W_fc1', shape=[4*4*220, 500], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc1 = tf.Variable(tf.constant(value=0.001, shape=[500], name='b_fc1'))
                self.L5_fc1 = tf.matmul(self.L4_con_layer, self.W_fc1) + self.b_fc1
                self.L5_fc1 = self.BN(input=self.L5_fc1, scale=True, training=self.training, name='L5_fc1')
                self.L5_fc1 = self.parametric_relu(self.L5_fc1, 'Ru_fc1')
                # self.L_fc1 = tf.layers.dropout(inputs=self.L_fc1, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## �� fully connected ���� - 2
            ##  �� ����ġ      �� shape: (625, 625), output: 625 ��, �ʱⰪ: He
            ##  �� ����        �� shape: 625, �ʱⰪ: 0.001
            ##  �� Ȱ��ȭ �Լ� �� Parametric Relu
            ##  �� ��� �ƿ� ����
            ############################################################################################################
            with tf.name_scope('fc_layer2') as scope:
                self.W_fc2 = tf.get_variable(name='W_fc2', shape=[500, 500], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc2 = tf.Variable(tf.constant(value=0.001, shape=[500], name='b_fc2'))
                self.L6_fc2 = tf.matmul(self.L5_fc1, self.W_fc2) + self.b_fc2
                self.L6_fc2 = self.BN(input=self.L6_fc2, scale=True, training=self.training, name='L6_fc2')
                self.L6_fc2 = self.parametric_relu(self.L6_fc2,'Ru_fc2')
                # self.L_fc2 = tf.layers.dropout(inputs=self.L_fc2, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## �� �����
            ##  �� ����ġ      �� shape: (625, 10), output: 2 ��, �ʱⰪ: He
            ##  �� ����        �� shape: 2, �ʱⰪ: 0.001
            ##  �� Ȱ��ȭ �Լ� �� Softmax
            ############################################################################################################
            self.W_out = tf.get_variable(name='W_out', shape=[500, 2], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[2], name='b_out'))
            self.logits = tf.matmul(self.L6_fc2, self.W_out) + self.b_out

        ################################################################################################################
        ## �� L2-Regularization
        ##  �� ��/(2*N)*��(W)��-> (0.001/(2*tf.to_float(tf.shape(self.X)[0])))*tf.reduce_sum(tf.square(self.W7))
        ################################################################################################################
        weight_decay = (0.01/(2*tf.to_float(tf.shape(self.Y)[0])))*tf.reduce_sum(tf.square(self.W_out))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

        # self.tensorflow_summary()

    # ####################################################################################################################
    # ## �� Tensorboard logging
    # ##  �� tf.summary.histogram : �������� ��� ���� logging �ϴ� ���
    # ##  �� tf.summary.scalar : �Ѱ��� ��� ���� logging �ϴ� ���
    # ####################################################################################################################
    # def tensorflow_summary(self):
    #     self.W1_hist = tf.summary.histogram('W1_conv1', self.W1)
    #     self.b1_hist = tf.summary.histogram('b1_conv1', self.b1)
    #     self.L1_hist = tf.summary.histogram('L1_conv1', self.L1)
    #
    #     self.W2_hist = tf.summary.histogram('W2_conv2', self.W2)
    #     self.b2_hist = tf.summary.histogram('b2_conv2', self.b2)
    #     self.L2_hist = tf.summary.histogram('L2_conv2', self.L2)
    #
    #     self.W3_hist = tf.summary.histogram('W3_conv3', self.W3)
    #     self.b3_hist = tf.summary.histogram('b3_conv3', self.b3)
    #     self.L3_hist = tf.summary.histogram('L3_conv3', self.L3)
    #
    #     self.W4_hist = tf.summary.histogram('W4_conv4', self.W4)
    #     self.b4_hist = tf.summary.histogram('b4_conv4', self.b4)
    #     self.L4_hist = tf.summary.histogram('L4_conv4', self.L4)
    #
    #     self.W5_hist = tf.summary.histogram('W5_conv5', self.W5)
    #     self.b5_hist = tf.summary.histogram('b5_conv5', self.b5)
    #     self.L5_hist = tf.summary.histogram('L5_conv5', self.L5)
    #
    #     self.W_fc1_hist = tf.summary.histogram('W6_fc1', self.W_fc1)
    #     self.b_fc1_hist = tf.summary.histogram('b6_fc1', self.b_fc1)
    #     self.L_fc1_hist = tf.summary.histogram('L6_fc1', self.L_fc1)
    #
    #     self.W_fc2_hist = tf.summary.histogram('W6_fc2', self.W_fc2)
    #     self.b_fc2_hist = tf.summary.histogram('b6_fc2', self.b_fc2)
    #     self.L_fc2_hist = tf.summary.histogram('L6_fc2', self.L_fc2)
    #
    #     self.cost_hist = tf.summary.scalar(self.name+'/cost_hist', self.cost)
    #     self.accuracy_hist = tf.summary.scalar(self.name+'/accuracy_hist', self.accuracy)
    #
    #     # �� merge_all �� �ϴ� ���, hist �� ������ �ʴ� �����鵵 ����� �Ǿ ������ �߻��Ѵ�.
    #     #    ���� merge �� ���������ϴ� ������ ���� ����������Ѵ�.
    #     self.merged = tf.summary.merge([self.W1_hist, self.b1_hist, self.L1_hist,
    #                                     self.W2_hist, self.b2_hist, self.L2_hist,
    #                                     self.W3_hist, self.b3_hist, self.L3_hist,
    #                                     self.W4_hist, self.b4_hist, self.L4_hist,
    #                                     self.W5_hist, self.b5_hist, self.L5_hist,
    #                                     self.W_fc1_hist, self.b_fc1_hist, self.L_fc1_hist,
    #                                     self.W_fc2_hist, self.b_fc2_hist, self.L_fc2_hist,
    #                                     self.cost_hist, self.accuracy_hist])

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

    ####################################################################################################################
    ## �� Parametric Relu or Leaky Relu
    ##  �� alpha ���� ������ 0 ������ ��� alpha ��ŭ�� ��縦 �����ؼ� 0 �� �ƴ� ���� �����ϴ� �Լ�
    ##  �� Parametric Relu : �н��� ���� ����ȭ�� alpha ���� ���ؼ� �����ϴ� Relu �Լ�
    ##  �� Leaky Relu      : 0 �� ������ ���� ���� alpha ������ �����ϴ� Relu �Լ�
    ####################################################################################################################
    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

    ####################################################################################################################
    ## �� Maxout - Created by ������
    ##  �� Convolution �����̳� FC �������� Ȱ��ȭ �Լ� ��� dropout �� ȿ���� �ش�ȭ�ϱ� ���� ����ϴ� �Լ�
    ##  �� conv �Ǵ� affine ������ ��ģ ���鿡 ���� k ���� �׷����� �����ϰ� �ش� �׷쳻���� ���� ū ���� ���� ��������
    ##     ������ ���
    ####################################################################################################################
    def max_out(self, inputs, num_units, axis=None):
        shape = inputs.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError(
                'number of features({}) is not a multiple of num_units({})'.format(num_channels, num_units))
        shape[axis] = num_units  # m
        shape += [num_channels // num_units]  # k
        outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
        return outputs

    ####################################################################################################################
    ## �� Batch Normalization - Created by ������,�ڻ��
    ##  �� training �ϴ� ���� ��ü�� ��ü������ ����ȭ�Ͽ� �н� �ӵ��� ���ӽ�ų �� �ִ� ���
    ##  �� Network�� �� ���̳� Activation ���� input_data �� distribution �� ��� 0, ǥ������ 1�� input_data�� ����ȭ��Ű�� ���
    ##  �� �ʱ� �Ķ���� --> beta : 0 , gamma : 1 , decay : 0.99 , epsilon : 0.001
    ####################################################################################################################
    def BN(self, input, training, scale, name, decay=0.99):
        return tf.contrib.layers.batch_norm(input, decay=decay, scale=scale, is_training=training, updates_collections=None, scope=name)

    ####################################################################################################################
    ## �� dynamic_learning - Created by ������
    ##  �� epoch �� Ŭ���� �ƴϸ� early_stopping �� ���۵Ǹ� ���������� learning_rate�� ���� �ٿ� �������� �Ʒ��� ������ ���
    ####################################################################################################################
    def dynamic_learning(self,learning_rate,earlystop,epoch):
        max_learning_rate = learning_rate
        min_learing_rate = 0.001
        learning_decay = 60 # �������� ���� ��������.
        if earlystop >= 1:
            lr = min_learing_rate + (max_learning_rate - min_learing_rate) * np.exp(-epoch / learning_decay)
        else:
            lr = max_learning_rate
        return round(lr,4)
