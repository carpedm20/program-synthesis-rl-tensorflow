import numpy as np
import tensorflow as tf

def encoder_fn(inputs, outputs):
    with tf.variable_scope("encoder"):
        input_enc = conv2d(inputs, 32, name="input")
        output_enc = conv2d(outputs, 32, name="output")

        conv_fn = lambda x, name: conv2d(x, 64, name=name)

        grid_enc = tf.concat([input_enc, output_enc], -1)
        res1 = residual_block(grid_enc, conv_fn, 3, "res1")
        res2 = residual_block(res1, conv_fn, 3, "res2")

    return linear(flatten(res2), 512)

def linear(inputs, output):
    return tf.layers.dense(inputs, output)

def flatten(inputs):
    shape = int_shape(inputs)
    last_dim = np.prod(shape[1:])
    return tf.reshape(inputs, [-1, last_dim], name="flat")

def residual_block(
        inputs, conv_fn, depth, name="res_block"):
    with tf.variable_scope(name):
        out = inputs
        for idx in range(depth):
            out = conv_fn(out, name="conv{}".format(idx))
        out += inputs
    return out

def conv2d(
        inputs,
        filters,
        kernel_size=(3, 3), 
        activation=tf.nn.relu,
        padding='same',
        name="conv2d"):
    out = tf.layers.conv2d(
            inputs,
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            name=name,
    )
    return out

def int_shape(x):
    return tuple(x.get_shape().as_list())
