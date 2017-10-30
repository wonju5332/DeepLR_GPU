import time
import tensorflow as tf
import numpy as np

# hyper parameters
learning_rate = 0.001
training_epochs = 20

# batch parameters
batch_size =100
train_num = 8000

# load data
print('Loading data....')
data = np.loadtxt('C:\\data3\\places3.csv', delimiter=',')
data = data[:10000]
np.random.shuffle(data)

# data setting -> norm and one-hot-encoding
def data_setting(data):
    x = (np.array(data[:, 0:-1]) / 255).tolist()
    y = [[1, 0] if y_ == 0 else [0, 1] for y_ in data[:, [-1]]]  # one_hot_encoding
    train_idx = int(round(len(data) * 0.8))
    return x[0:train_idx], y[0:train_idx], x[train_idx:len(data)], y[train_idx:len(data)]

x_train, t_train, x_test, t_test = data_setting(data)

# np array로 바꾸기
x_train, t_train = np.array(x_train), np.array(t_train)
x_test, t_test = np.array(x_test), np.array(t_test)

print('Train num',x_train.shape[0]) # 8000
print('Data of shape', data.shape)  # 10000, 6401
print('Train label one hot encoding', t_train[0]) # [0,1]
print('Loading and data setting done!')

# initializer
init = tf.global_variables_initializer()

# batch normalization
def BN(input, training, name, scale=True, decay=0.99):
    return tf.contrib.layers.batch_norm(input, decay=decay, scale=scale, is_training=training, updates_collections=None,
                                        scope=name)

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, 6400]) # 80*80
            X_img = tf.reshape(self.X, [-1, 80, 80, 1])  # channel = grayscale
            self.Y = tf.placeholder(tf.float32, [None, 2])

            # Convolutional Layer #1 and  Pooling Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=None)
            conv1_bn = BN(input=conv1, training=1, name='conv1_bn')
            conv1_bn_rl = tf.nn.relu(conv1_bn, name='conv1_bn_rl')
            pool1 = tf.layers.max_pooling2d(inputs=conv1_bn_rl, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=tf.nn.relu)
            conv2_bn = BN(input=conv2, training=1, name='conv2_bn')
            conv2_bn_rl = tf.nn.relu(conv2_bn, name='conv2_bn_rl')
            pool2 = tf.layers.max_pooling2d(inputs=conv2_bn_rl, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=tf.nn.relu)
            conv3_bn = BN(input=conv3, training=1, name='conv3_bn')
            conv3_bn_rl = tf.nn.relu(conv3_bn, name='conv3_bn_rl')
            pool3 = tf.layers.max_pooling2d(inputs=conv3_bn_rl, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            # Convolutional Layer #4 and Pooling Layer #4
            conv4 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=tf.nn.relu)
            conv4_bn = BN(input=conv4, training=1, name='conv4_bn')
            conv4_bn_rl = tf.nn.relu(conv4_bn, name='conv4_bn_rl')
            pool4 = tf.layers.max_pooling2d(inputs=conv4_bn_rl, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            print(pool4.shape)  # 5*5*256
            # Dense Layer with Relu
            flat = tf.reshape(pool4, [-1, 5 * 5 * 256])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=1024, activation=tf.nn.relu)
            dropout5 = tf.layers.dropout(inputs=dense4,
                                         rate=0.5, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout5, units=2)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def etrain(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

tf.reset_default_graph()
sess = tf.Session()
models = []
num_models = 5
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Started!')

t1=time.time()
# train my model
for epoch in range(training_epochs):

    avg_cost_list = np.zeros(len(models))
    total_batch = int( train_num / batch_size)

    # print('total_batch', total_batch)
    for step in range(0, train_num, batch_size):
        batch_xs, batch_ys = x_train[step:step + batch_size], t_train[step:step + batch_size]
        # print('batch_xs.shape', batch_xs.shape)
        # print('batch_xs.type', batch_xs.type)
        # train each model

        for m_idx, m in enumerate(models):
            c, _ = m.etrain(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

print('Learning Finished!')

# Test model and check accuracy
# train_num = len(mnist.test.labels)
print('Test Started!')
print('Train_num', x_train.shape[0])

predictions = np.zeros(len(x_test) * 2).reshape(len(x_test), 2)

test_len = len(x_test)

for step in range(0, test_len, batch_size):
    model_result = np.zeros(batch_size, dtype='int32')
    for m_idx, m in enumerate(models):
        print(m_idx, 'Accuracy:', m.get_accuracy(x_test[step:step + batch_size], t_test[step:step + batch_size]))
        p = m.predict(x_test[step:step + batch_size])
        # model_result += np.argmax(p,1)
        model_result[:] += np.argmax(p, 1)

        for idx, result in enumerate(model_result):
            predictions[idx, result] += 1

ensemble_correct_prediction = tf.equal(
    tf.argmax(predictions, 1), tf.argmax(t_test, 1))
ensemble_accuracy = tf.reduce_mean(
    tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

t2=time.time()
print("Total running time : %s sec" %(t2-t1))