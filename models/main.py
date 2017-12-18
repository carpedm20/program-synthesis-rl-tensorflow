import tensorflow as tf

from .decoder import Decoder
from .encoder import encoder_fn

class Model(object):
    def __init__(self, config, inputs, outputs, codes, code_lengths, dataset):
        self.config = config

        # [BxN, 512]
        encoder_out = encoder_fn(inputs, outputs, codes, dataset)
        decoder = Decoder(config, codes, encoder_out, code_lengths, dataset)

        self.mle_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=decoder,
                logits=decoder.logits,
                scope="MLE_loss")

        self.optimizer = tf.train.AdamOptimizer(config.lr)

        if self.use_rl:
            self.loss = None
        else:
            self.loss = self.mle_loss

        self.optim = self.optimizer.optimize(self.loss)

    def update(self, sess):
        sess.run(self.optim)
