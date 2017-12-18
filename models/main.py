import tensorflow as tf

from .decoder import Decoder
from .encoder import encoder_fn

class Model(object):
    def __init__(self, config, inputs, outputs, codes, code_lengths, dataset):
        self.config = config

        # [BxN, 512]
        encoder_out = encoder_fn(inputs, outputs, codes, dataset)
        decoder = Decoder(config, codes, encoder_out, code_lengths, dataset)

        self.mle_loss = tf.contrib.seq2seq.sequence_loss(
                logits=decoder.logits,
                targets=codes,
                weights=tf.sequence_mask(code_lengths, dtype=tf.float32),
                name="MLE_loss")

        if config.use_rl:
            self.loss = None
        else:
            self.loss = self.mle_loss

        self.global_step = tf.train.get_or_create_global_step()
        self.optim = tf.train.AdamOptimizer(config.lr) \
                             .minimize(self.loss, self.global_step)

    def build_summary(self, sess):
        tf.summary.scalar(self.mle_loss, "mle_loss")
        tf.summary.scalar(self.loss, "loss")

    def update(self, sess):
        sess.run(self.optim)
