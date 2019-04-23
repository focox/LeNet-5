import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import inference
import train
import numpy as np


def evaluate(mnist):
    with tf.Graph().as_default() as g:

        xs = mnist.validation.images
        xs = np.reshape(xs, [-1, inference.IMAGE_SIZE, inference.IMAGE_SIZE, inference.NUM_CHANNELS])
        ys = mnist.validation.labels

        num_validation = np.shape(ys)[0]

        x = tf.placeholder(tf.float32, [num_validation, inference.IMAGE_SIZE, inference.IMAGE_SIZE, inference.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [num_validation, inference.OUTPUT_NODE])
        y = inference.infer(x, False, None)

        accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy_op = tf.reduce_mean(tf.cast(accuracy, tf.float32))


        average_variables = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        average_variables_restore = average_variables.variables_to_restore()

        ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(average_variables_restore)

            with tf.Session() as sess:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy_op, feed_dict={x: xs, y_: ys})
                print('After %s step(s), the accuracy on validation is %g' % (global_step, accuracy_score))
        else:
            print('No checkpoing file found')
            return


def main(argv=None):
    mnist = input_data.read_data_sets('./mnist_data/', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()