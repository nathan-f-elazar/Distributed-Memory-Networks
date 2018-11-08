"""
This file contains utility functions for quickly defining tensorflow graphs
"""
import tensorflow as tf


def _linear(x, name, out_dim, bias=True, bias_init=None, weight_init=None):
    """
    Applies an right affine transformation to the input variable x.
    :param name: scope name to differentiate internal variables.
    :param bias_init: an initializer that will be used for the bias.
    :param weight_init: an initializer that will be used for the bias.
    :return: [x.shape[0], x.shape[1], out_dim] tensor that is the result of the affine transformation.
    """
    with tf.variable_scope(name) as scope:
        try:
            if bias_init is None:
                W = tf.get_variable('W', [x.shape[-1], out_dim])
            else:
                W = tf.get_variable('W', [x.shape[-1], out_dim], initializer=tf.constant_initializer(weight_init))
            if bias_init is None:
                b = tf.get_variable('b', [1, out_dim])
            else:
                b = tf.get_variable('b', [1, out_dim], initializer=tf.constant_initializer(bias_init))
        except ValueError:
            scope.reuse_variables()
            W = tf.get_variable('W')
            b = tf.get_variable('b')
    return tf.matmul(x, W) + b if bias else tf.matmul(x, W)


def _linearX(x, name, out_dim, bias=True, bias_init=None, weight_init=None):
    """
    similar to _linear except this function takes a list of inputs and applies
    affine transformations to each element of the input.
    """
    outs = []
    with tf.variable_scope(name) as scope:
        for i in range(len(x)):
            try:
                if bias_init is None:
                    W = tf.get_variable('W'+str(i), [x[i].shape[-1], out_dim])
                else:
                    W = tf.get_variable('W'+str(i), [x[i].shape[-1], out_dim], initializer=tf.constant_initializer(weight_init))
                if bias_init is None:
                    b = tf.get_variable('b'+str(i), [1, out_dim])
                else:
                    b = tf.get_variable('b'+str(i), [1, out_dim], initializer=tf.constant_initializer(bias_init))
            except ValueError:
                scope.reuse_variables()
                W = tf.get_variable('W'+str(i))
                b = tf.get_variable('b'+str(i))
            outs.append(tf.matmul(x[i], W) + b if bias else tf.matmul(x[i], W))
    return tf.add_n(outs)


def _embed(x, name, n_items, dims, init=None):
    """
    Creates an embedding of x.
    :param x: integer ID to be embedded.
    :param n_items: size of the embedding table, indexes above n_items will raise an error.
    :param dims: embedding dimensions .
    :return: [x.shape[0], dims[0], ..., dims[-1]] tensor embedding of x.
    """
    with tf.variable_scope(name) as scope:
        try:
            if init is None:
                E = tf.get_variable('E', [n_items] + dims)
            else:
                E = tf.get_variable('E', [n_items] + dims, initializer=init)
        except ValueError:
            scope.reuse_variables()
            E = tf.get_variable('E')
    return tf.gather(E, x)


def _gates(x, name, n_gates=1):
    """
    Applies a linear transformation then a sigmoid activation, n_gates times.
    This is used by recurrent neural network models to create gate outputs.
    :param x: input tensor.
    :param n_gates: number of gate tensors to produce.
    :return: [x.shape[0], x.shape[1], x.shape[-1], n_gates] tensor of gate outputs.
    """
    with tf.variable_scope(name):
        return tf.split(tf.nn.sigmoid(_linearX(x, 'gates', x[0].shape[-1].value * n_gates)), n_gates, -1)
