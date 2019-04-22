import tensorflow as tf

BATCH_SIZE = 100
REGULARIZER_RATE = 0.01

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

MOVING_AVERAGE_DECAY = 0.999


def train():
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y-output')

    