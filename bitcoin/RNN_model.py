import tensorflow as tf
from bitcoin import BNLSTMCell

class RNN_Model:
    def __init__(self, sess, n_inputs, n_sequences, n_hiddens, n_outputs, hidden_layer_cnt, file_name, model_name):
        self.sess = sess
        self.n_inputs = n_inputs
        self.n_sequences = n_sequences
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.hidden_layer_cnt = hidden_layer_cnt
        self.file_name = file_name
        self.model_name = model_name
        self.regularizer = tf.contrib.layers.l2_regularizer(0.001)
        self.training = True
        self._build_net()

    def _build_net(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope(self.model_name):
                self.learning_rate = 0.001

                self.X = tf.placeholder(tf.float32, [None, self.n_sequences, self.n_inputs])
                self.Y = tf.placeholder(tf.float32, [None, self.n_outputs])

                self.multi_cells = tf.contrib.rnn.MultiRNNCell([self.lstm_cell(self.n_hiddens) for _ in range(self.hidden_layer_cnt)], state_is_tuple=True)
                self.outputs, _states = tf.nn.dynamic_rnn(self.multi_cells, self.X, dtype=tf.float32)

                # self.outputs = tf.reshape(self.outputs, shape=[-1, self.n_sequences * self.n_hiddens])
                # self.fc1 = tf.contrib.layers.fully_connected(self.outputs, 200)
                # self.Y_ = tf.contrib.layers.fully_connected(self.fc1, self.n_outputs, activation_fn=None)
                self.Y_ = tf.contrib.layers.fully_connected(self.outputs[:, -1], self.n_outputs, activation_fn=None)
                self.reg_loss = tf.reduce_sum([self.regularizer(train_var) for train_var in tf.trainable_variables() if re.search('(kernel)|(weights)', train_var.name) is not None])
                self.loss = self.huber_loss(0.5 * tf.reduce_sum(tf.square(self.Y_ - self.Y)) + self.reg_loss)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

                self.targets = tf.placeholder(tf.float32, [None, 1])
                self.predictions = tf.placeholder(tf.float32, [None, 1])
                self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))

    def huber_loss(self, loss):
        return tf.where(tf.abs(loss) <= 1.0, 0.5 * tf.square(loss), tf.abs(loss) - 0.5)

    def lstm_cell(self, hidden_size):
        cell = BNLSTMCell(hidden_size, self.training)
        # cell = BNGRUCell(hidden_size, self.training)
        # cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
        # if self.training:
        #     cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5)
        return cell

    def train(self, x_data, y_data):
        self.training = True
        return self.sess.run([self.reg_loss, self.loss, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data})

    def predict(self, x_data):
        self.training = False
        return self.sess.run(self.Y_, feed_dict={self.X: x_data})

    def rmse_predict(self, targets, predictions):
        self.training = False
        return self.sess.run(self.rmse, feed_dict={self.targets: targets, self.predictions: predictions})
