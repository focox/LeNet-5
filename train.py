import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import inference
import os
import numpy as np
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'model.ckpt'

BATCH_SIZE = 100
TRAINING_STEPS = 30001

REGULARIZER_RATE = 0.0001

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99

MOVING_AVERAGE_DECAY = 0.99


def train(mnist, train=True):
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, inference.IMAGE_SIZE, inference.IMAGE_SIZE, inference.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, inference.OUTPUT_NODE], name='y-output')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    y = inference.infer(x, train, regularizer)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    losses = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    global_step = tf.Variable(0, trainable=False)

    average_variables = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    average_variables_op = average_variables.apply(tf.trainable_variables())

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(losses, global_step=global_step)

    # train_op = tf.group(train_step, average_variables_op)

    with tf.control_dependencies([train_step, average_variables_op]):
        train_op = tf.no_op('train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            xs = np.reshape(xs, [BATCH_SIZE, inference.IMAGE_SIZE, inference.IMAGE_SIZE, inference.NUM_CHANNELS])
            loss_value, steps, _ = sess.run([losses, global_step, train_op], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print('After %d step(s), the loss on training is %s' % (steps, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step)


def main(argv=None):
    mnist = input_data.read_data_sets('./mnist_data/', one_hot=True)
    train(mnist, True)


if __name__ == '__main__':
    tf.app.run()