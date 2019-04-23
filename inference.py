import tensorflow as tf
INPUT_NODE = 784
IMAGE_SIZE = 28
OUTPUT_NODE = 10
NUM_CHANNELS = 1

# layer1 convolution
CONV1_DEEP = 32
CONV1_SIZE = 5

# layer2 convolution
CONV2_DEEP = 64
CONV2_SIZE = 5

# full connection
FC_SIZE = 512


def get_weights(shape, regularizer):
    weights = tf.get_variable('weights', shape, tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    if regularizer:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def get_biases(shape, value):
    biases = tf.get_variable('biases', shape, tf.float32, tf.constant_initializer(value))
    return biases


def infer(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-convolution'):
        # 只有全连接需要加入正则化
        weights = get_weights([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], None)
        conv = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding='SAME')
        biases = get_biases([CONV1_DEEP], 0.0)
        layer1 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        # 28X28X1 => 28X28X32

    with tf.variable_scope('layer2-pooling'):
        layer2 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 14X14X32

    with tf.variable_scope('layer3-convolution'):
        # 只有全连接需要加入正则化
        weights = get_weights([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], None)
        biases = get_biases([CONV2_DEEP], 0.0)
        conv = tf.nn.conv2d(layer2, weights, strides=[1, 1, 1, 1], padding='SAME')
        layer3 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        # 14X14X64

    with tf.variable_scope('layer4-pooling'):
        layer4 = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 7x7x64

    pool_shape = layer4.get_shape().as_list()
    # pool_shape[0] 为一个BATCH_SIZE数据的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(layer4, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-dense'):
        weights = get_weights(shape=[nodes, FC_SIZE], regularizer=regularizer)
        biases = get_biases([FC_SIZE], 0.1)
        layer5 = tf.nn.relu(tf.matmul(reshaped, weights) + biases)
        if train:
            layer5 = tf.nn.dropout(layer5, 0.5)

    with tf.variable_scope('layer6-dense'):
        weights = get_weights([FC_SIZE, OUTPUT_NODE], regularizer)
        biases = get_biases([OUTPUT_NODE], 0.1)
        layer6 = tf.matmul(layer5, weights) + biases

    return layer6