import tensorflow as tf
from tensorflow.layers import conv2d, Flatten, Dense, conv2d_transpose
from tensorflow.contrib.layers import batch_norm

def _leaky_relu(x, alpha=0.2):
    return tf.maximum(x, tf.multiply(x, alpha))

def _conv2d(input, name, filters=64, kernel_size=4, strides=2, padding="same"):
    with tf.variable_scope(name):
        return conv2d(inputs=input, name=name, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)

def _conv2d_transpose(input, name, filters=64, kernel_size=4, strides=2, padding="same"):
    with tf.variable_scope(name):
        return conv2d_transpose(inputs=input, name=name, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)

def _batch_norm(input, name, is_training, momentum=0.99, epsilon=1e-3):
    # Batch Normalization
    # Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    # This function spectral_norm is forked from "https://github.com/MingtaoGuo/ContextEncoder_Cat-s_head_Inpainting_TensorFlow"
    with tf.variable_scope(name):
        beta = tf.Variable(tf.constant(0.0, shape=[input.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[input.shape[-1]]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
    return normed

def _fully_connected_2d(input, name):
    previous_size = input.shape
    with tf.variable_scope(name):
        input = Flatten()(input)
        input = Dense(1000)(input)
        input = Dense(int(previous_size[1] * previous_size[2] * previous_size[3]))(input)
        input = tf.reshape(input, (-1, int(previous_size[1]), int(previous_size[2]), int(previous_size[3])))
        return input

def _fully_connected(input, name):
    with tf.variable_scope(name):
        input = Flatten()(input)
        input = Dense(1)(input)
        return input
