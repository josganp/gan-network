"""Module for tensorboard ops.
From TensorFlow Tutorial see
https://www.tensorflow.org/get_started/summaries_and_tensorboard
"""

import tensorflow as tf


def variable_summaries(var):
    """Attatch summaries of a variable to a Tensor for TensorBoard.

    Args:
        var (tf.Tensor): Tensor variable.
    """

    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.compat.v1.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
        tf.compat.v1.summary.histogram('histogram', var)
