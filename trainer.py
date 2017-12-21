import os
from tqdm import trange
import tensorflow as tf
from models import encoder, decoder

from models import Model
from dataset import KarelDataset

class Trainer(object):
    def __init__(self, config, rng=None):
        self.config = config
        self.dataset = KarelDataset(config, rng)

        self.global_step = tf.train.get_or_create_global_step()

        self.train_model = Model(self.dataset, self.config, self.global_step, train=True)
        self.train_model.build_optim()

    def train(self, sess):
        sess.run(tf.global_variables_initializer())
        sess.run(self.train_model.iterator.initializer)

        for _ in trange(self.config.max_step):
            step = self.train_model.run(sess)

            if step % self.config.log_step == 0:
                step = self.train_model.run(
                        sess, with_update=False, with_summary=True)

    def test(self):
        pass
