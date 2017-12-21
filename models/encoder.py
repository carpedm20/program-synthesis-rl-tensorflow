import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell

def encoder(inputs, outputs, codes, dataset):
    input_shape = int_shape(inputs)
    output_shape = int_shape(outputs)

    # make examples to batch. batch_size := num_examples * batch_size
    flat_inputs = tf.reshape(inputs, [-1] + input_shape[2:])
    flat_outputs = tf.reshape(outputs, [-1] + output_shape[2:])

    with tf.variable_scope("encoder"):
        input_enc = conv2d(flat_inputs, 32, name="input")
        output_enc = conv2d(flat_outputs, 32, name="output")

        conv_fn = lambda x, name: conv2d(x, 64, name=name)

        grid_enc = tf.concat([input_enc, output_enc], -1)
        res1 = residual_block(grid_enc, conv_fn, 3, "res1")
        res2 = residual_block(res1, conv_fn, 3, "res2")

        # [BxN, 512]
        cnn_out = linear(flatten(res2), 512, "cnn_out")

    return cnn_out

def linear(x, dim, name):
    return tf.layers.dense(x, dim, name=name)

def flatten(x):
    shape = int_shape(x)
    last_dim = np.prod(shape[1:])
    return tf.reshape(x, [-1, last_dim], name="flat")

def residual_block(
        x, conv_fn, depth, name="res_block"):
    with tf.variable_scope(name):
        out = x
        for idx in range(depth):
            out = conv_fn(out, name="conv{}".format(idx))
        out += x
    return out

def conv2d(
        x,
        filters,
        kernel_size=(3, 3), 
        activation=tf.nn.relu,
        padding='same',
        name="conv2d"):
    out = tf.layers.conv2d(
            x,
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            name=name,
    )
    return out

def int_shape(x):
    return list(x.get_shape().as_list())
