import tensorflow as tf

def encoder_fn(inputs, outputs):
    input_enc = cnn_encoder(inputs, scope="input")
    output_enc = cnn_encoder(outputs, scope="output", reuse=True)
    return input_enc, output_enc

def cnn_encoder(inputs, scope, reuse=False):
    with tf.variable_scope("{}_embed".format(scope)):
        enc = conv2d(inputs, 32)

    with tf.variable_scope("{}_res".format(scope), reuse=reuse):
        conv_fn = lambda inputs, name: conv2d(inputs, 64, name)
        res1 = residual_block(grid_enc, conv_fn, 3, "res1")
        res2 = residual_block(res1, conv_fn, 3, "res2")

    return linear(flatten(res2), 512)

def linear(inputs, output):
    return tf.layers.linear(inputs, output)

def flatten(inputs, batch_size):
    name = "flat_{}".format(inputs.name)
    return tf.reshape(inputs, [batch_size, -1], name)

def res_block(
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
