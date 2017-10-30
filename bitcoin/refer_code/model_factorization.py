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
                # self.dropout_rate = tf.Variable(tf.constant(value=0.5), name='dropout_rate')
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

            with tf.name_scope('conv_sub_1') as scope:
                self.W1_sub1_1 = tf.get_variable(name='W1_sub1_1', shape=[1, 3, 3, 40], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
                # self.b1_sub1_1 = tf.Variable(tf.constant(value=0.001, shape=[40]), name='b1_sub1_1')
                self.L1_sub1_1 = tf.nn.conv2d(name='L1_con_sub_1_1',input=self.X, filter=self.W1_sub1_1, strides=[1, 1, 1, 1], padding='VALID')  # 50x50 -> 50x48
                self.L1_sub1_1 = self.BN(input=self.L1_sub1_1, scale=True, training=self.training,
                                         name='L1_BN_sub_1_1')
                self.L1_sub1_1 = self.parametric_relu(self.L1_sub1_1, 'R1_sub1_1')



                self.W1_sub1_2 = tf.get_variable(name='W1_sub1_2', shape=[3, 1, 40, 40], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L1_sub1_2 = tf.nn.conv2d(name='L1_con_sub1_2',input=self.L1_sub1_1, filter=self.W1_sub1_2, strides=[1, 1, 1, 1], padding='VALID')  # 50x48 -> 48x48
                self.L1_sub1_2 = self.BN(input=self.L1_sub1_2, scale=True, training=self.training,
                                             name='L1_BN_sub_1_2')
                self.L1_sub1_2 = self.parametric_relu(self.L1_sub1_2, 'R1_sub1_2')


                self.W1_sub1_3 = tf.get_variable(name='W1_sub1_3', shape=[1, 3, 40, 40], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
                # self.b1_sub1_3 = tf.Variable(tf.constant(value=0.001, shape=[40]), name='b1_sub1_3')
                self.L1_sub1_3 = tf.nn.conv2d(name='L1_con_sub_1_3',input=self.L1_sub1_2, filter=self.W1_sub1_3, strides=[1, 1, 1, 1], padding='VALID')  # 48x48 ->48x46
                self.L1_sub1_3 = self.BN(input=self.L1_sub1_3, scale=True, training=self.training,
                                             name='L1_BN_sub_1_3')
                self.L1_sub1_3 = self.parametric_relu(self.L1_sub1_3, 'R1_sub1_3')

                self.W1_sub1_4 = tf.get_variable(name='W1_sub1_4', shape=[3, 1, 40, 40], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
                # self.b1_sub1_4 = tf.Variable(tf.constant(value=0.001, shape=[40]), name='b1_sub1_4')
                self.L1_sub1_4 = tf.nn.conv2d(name='L1_con_sub_1_4',input=self.L1_sub1_3, filter=self.W1_sub1_4, strides=[1, 1, 1, 1], padding='VALID')  # 48x46 -> 46x46
                self.L1_sub1_4 = self.BN(input=self.L1_sub1_4, scale=True, training=self.training,
                                             name='L1_sub_BN_1_4')
                self.L1_sub1_4 = self.parametric_relu(self.L1_sub1_4, 'R1_sub1_4')



            with tf.name_scope('conv_sub_2') as scope:
                self.W1_sub2_1 = tf.get_variable(name='W1_sub2_1', shape=[1, 3, 40, 80], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L1_sub2_1 = tf.nn.conv2d(input=self.L1_sub1_4, filter=self.W1_sub2_1, strides=[1, 1, 1, 1], padding='VALID')  # 46x46->46x44
                self.L1_sub2_1 = self.BN(input=self.L1_sub2_1, scale=True, training=self.training,
                                         name='L1_sub_BN_2_1')
                self.L1_sub2_1 = self.parametric_relu(self.L1_sub2_1, 'R1_sub2_1')



                self.W1_sub2_2 = tf.get_variable(name='W1_sub2_2', shape=[3, 1, 80, 80], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
                # self.b1_sub2_2 = tf.Variable(tf.constant(value=0.001, shape=[80]), name='b1_sub2_2')
                self.L1_sub2_2 = tf.nn.conv2d(input=self.L1_sub2_1, filter=self.W1_sub2_2, strides=[1, 1, 1, 1], padding='VALID')  # 46x44 -> 44x44
                self.L1_sub2_2 = self.BN(input=self.L1_sub2_2, scale=True, training=self.training,
                                         name='L1_sub_BN_2_2')
                self.L1_sub2_2 = self.parametric_relu(self.L1_sub2_2, 'R1_sub2_2')



                self.W1_sub2_3 = tf.get_variable(name='W1_sub2_3', shape=[1, 3, 80, 80], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
                # self.b1_sub2_3 = tf.Variable(tf.constant(value=0.001, shape=[80]), name='b1_sub2_3')
                self.L1_sub2_3 = tf.nn.conv2d(input=self.L1_sub2_2, filter=self.W1_sub2_3, strides=[1, 1, 1, 1], padding='VALID')  # 44x44 ->44x42
                self.L1_sub2_3 = self.BN(input=self.L1_sub2_3, scale=True, training=self.training,
                                         name='L1_sub_BN_2_3')
                self.L1_sub2_3 = self.parametric_relu(self.L1_sub2_3, 'R1_sub2_3')



                self.W1_sub2_4 = tf.get_variable(name='W1_sub2_4', shape=[3, 1, 80, 80], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
                # self.b1_sub2_4 = tf.Variable(tf.constant(value=0.001, shape=[80]), name='b1_sub4')
                self.L1_sub2_4 = tf.nn.conv2d(input=self.L1_sub2_3, filter=self.W1_sub2_4, strides=[1, 1, 1, 1], padding='VALID')  # 44x42 -> 42x42
                self.L1_sub2_4 = self.BN(input=self.L1_sub2_4, scale=True, training=self.training,
                                         name='L1_sub_BN_2_4')
                self.L1_sub2_4 = self.parametric_relu(self.L1_sub2_4, 'R1_sub2_4')
                self.L1_sub2_4 = tf.nn.max_pool(value=self.L1_sub2_4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')  # 42X42 -> 21X21


            with tf.name_scope('conv_layer1') as scope:
                self.W2_sub1_1 = tf.get_variable(name='W2_sub1_1', shape=[1, 3, 80, 140], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L2_sub1_1 = tf.nn.conv2d(input=self.L1_sub2_4, filter=self.W2_sub1_1, strides=[1, 1, 1, 1],
                                              name='L2_con_sub1_1',padding='VALID')   # 21X21 -> 21X19
                self.L2_sub1_1 = self.BN(input=self.L2_sub1_1, scale=True, training=self.training,
                                         name='L2_sub_BN_1_1')
                self.L2_sub1_1 = self.parametric_relu(self.L2_sub1_1, 'L2_R1_sub1_1')


                self.W2_sub1_2 = tf.get_variable(name='W2_sub1_2', shape=[3, 1, 140, 140], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L2_sub1_2 = tf.nn.conv2d(input=self.L2_sub1_1, filter=self.W2_sub1_2, strides=[1, 1, 1, 1],
                                              name='L2_con_sub_1_2',padding='VALID')  #21X19 -> 19X19
                self.L2_sub1_2 = self.BN(input=self.L2_sub1_2, scale=True, training=self.training,
                                         name='L2_sub_BN_1_2')
                self.L2_sub1_2 = self.parametric_relu(self.L2_sub1_2, 'L2_R1_sub1_2')
                self.L2_sub1_2 = tf.nn.max_pool(value=self.L2_sub1_2,ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME') # 19X19 -> 10X10



            with tf.name_scope('conv_layer2') as scope:
                self.W3_sub1_1 = tf.get_variable(name='W3_sub1_1', shape=[1, 3, 140, 260], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L3_sub1_1 = tf.nn.conv2d(input=self.L2_sub1_2, filter=self.W3_sub1_1, strides=[1, 1, 1, 1],
                                              name='L3_con_sub1_1',padding='VALID')  #10x10 -> 10X8
                self.L3_sub1_1 = self.BN(input=self.L3_sub1_1, scale=True, training=self.training,
                                         name='L3_sub_BN_1_1')
                self.L3_sub1_1 = self.parametric_relu(self.L3_sub1_1, 'L3_R1_sub1_1')


                self.W3_sub1_2 = tf.get_variable(name='W3_sub1_2', shape=[3, 1, 260, 260], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L3_sub1_2 = tf.nn.conv2d(input=self.L3_sub1_1, filter=self.W3_sub1_2, strides=[1, 1, 1, 1],
                                              name='L3_con_sub_1_2',padding='VALID')  #10X8 -> 8X8
                self.L3_sub1_2 = self.BN(input=self.L3_sub1_2, scale=True, training=self.training,
                                         name='L3_sub_BN_1_2')
                self.L3_sub1_2 = self.parametric_relu(self.L3_sub1_2, 'L3_R1_sub1_2')
                self.L3_sub1_2 = tf.nn.max_pool(value=self.L3_sub1_2,ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME') #8X8 -> 4X4
                self.L3_con_layer = tf.reshape(self.L3_sub1_2, [-1, 4 * 4 * 260])


            ############################################################################################################
            ## �� fully connected ���� - 1
            ##  �� ����ġ      �� shape: (4 * 4 * 320, 625), output: 625 ��, �ʱⰪ: He
            ##  �� ����        �� shape: 625, �ʱⰪ: 0.001
            ##  �� Ȱ��ȭ �Լ� �� Parametric Relu
            ##  �� ��� �ƿ� ����
            ############################################################################################################
            with tf.name_scope('fc_layer1') as scope:
                self.W_fc1 = tf.get_variable(name='W_fc1', shape=[4*4*260, 500], dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc1 = tf.Variable(tf.constant(value=0.001, shape=[500], name='b_fc1'))
                self.L5_fc1 = tf.matmul(self.L3_con_layer, self.W_fc1) + self.b_fc1
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
                self.W_fc2 = tf.get_variable(name='W_fc2', shape=[500, 600], dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc2 = tf.Variable(tf.constant(value=0.001, shape=[600], name='b_fc2'))
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
            self.W_out = tf.get_variable(name='W_out', shape=[600, 2], dtype=tf.float32,
                                         initializer=tf.contrib.layers.variance_scaling_initializer())
            # lr = self.dynamic_learning(learning_rate=0.01,earlystop=self.early_stop_count,epoch=self.epoch)
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[2], name='b_out'))
            self.logits = tf.matmul(self.L6_fc2, self.W_out) + self.b_out

        ################################################################################################################
        ## �� L2-Regularization
        ##  �� ��/(2*N)*��(W)��-> (0.001/(2*tf.to_float(tf.shape(self.X)[0])))*tf.reduce_sum(tf.square(self.W7))
        ################################################################################################################

        weight_decay = (0.01/(2*tf.to_float(tf.shape(self.Y)[0])))*tf.reduce_sum(tf.square(self.W_out))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

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
