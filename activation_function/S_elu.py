import numpy as np
from tensorflow.python.framework import ops
import tensorflow as tf





def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))



with tf.Session() as sess:
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    a = sess.run(W)
    print(a)
