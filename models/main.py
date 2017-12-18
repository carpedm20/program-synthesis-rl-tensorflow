import tensorflow as tf

from .decoder import Decoder
from .encoder import encoder_fn

class Model(object):
    def __init__(self, config, inputs, outputs, codes, code_lengths, dataset):
        self.config = config

        # [BxN, 512]
        encoder_out = encoder_fn(inputs, outputs, codes, dataset)

        decoder = Decoder(config, codes, encoder_out, code_lengths, dataset)
        import ipdb; ipdb.set_trace() 

        self.optim = tf.train.AdamOptimizer(config.lr)

    def update(self, sess):
        sess.run(self.optim)
