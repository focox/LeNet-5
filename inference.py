import tensorflow as tf


def get_weights(shape, regularizer):
    weights = tf.get_variable('weights', shape, tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    if regularizer:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def get_biases(shape):
    biases = tf.get_variable('biases', shape, tf.float32, tf.constant_initializer(0.01))
    return biases


def infer(input_tensor, regularizer):
    with tf.variable_scope('layer1-convolution'):
        weights = get_weights(shape=[5, 5, 1, 6], regularizer=regularizer)
        biases = get_biases([6])
        layer1 = tf.nn.relu(tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding='VALID') + tf.nn.bias_add(biases))
        # 28x28x6

    with tf.variable_scope('layer2-pooling'):
        layer2 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # 14x14x6

    with tf.variable_scope('layer3-convolution'):
        weights = get_weights([5, 5, 6, 16], regularizer=regularizer)
        biases = get_biases([16])
        layer3 = tf.nn.relu(tf.nn.conv2d(layer2, weights, strides=[1, 1, 1, 1], padding='VALID') + tf.nn.bias_add(biases))
        # 10x10x16

    with tf.variable_scope('layer4-pooling'):
        layer4 = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # 5x5x16

    with tf.variable_scope('layer5-dense'):
        layer4_flatten = tf.layers.Flatten(layer4)
        weights = get_weights(shape=[layer4_flatten.shape[-1], 120], regularizer=regularizer)
        biases = get_biases([120])
        layer5 = tf.nn.relu(tf.matmul(layer4_flatten, weights) + biases)
        # None x 120

    with tf.variable_scope('layer6-dense'):
        weights = get_weights([120, 84], regularizer)
        biases = get_biases([84])
        layer6 = tf.nn.relu(tf.matmul(layer5, weights) + biases)
        # None x 84

    with tf.variable_scope('layer7-dense_output'):
        weights = get_weights([84, 10], regularizer)
        biases = get_biases([10])
        layer7 = tf.matmul(layer6, weights) + biases

    return layer7