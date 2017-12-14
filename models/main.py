from .decoder import Decoder
from .encoder import encoder_fn

class Model(object):
    def __init__(self, config, inputs, outputs, codes, dataset):
        self.config = config

        self.input_enc, self.output_enc = encoder_fn(inputs, outputs)
        self.decoder = Decoder(self.input_enc, self.output_enc, codes)

        self.optim = tf.train.AdamOptimizer(config.lr)

    def update(self, sess):
        sess.run(self.optim)
