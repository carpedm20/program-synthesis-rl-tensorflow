import os
import logging
from tqdm import trange
import tensorflow as tf

from models import Model
from dataset import KarelDataset
from models import encoder, decoder

logger = logging.getLogger("main")


class Trainer(object):
    def __init__(self, config, rng=None):
        self.config = config

        self.train_dataset = KarelDataset('train', config, rng)
        self.test_dataset = KarelDataset('test', config, rng)

        self.global_step = tf.train.get_or_create_global_step()

        self.train_model = Model(
                self.train_dataset, self.config,
                self.global_step, mode='train')
        self.train_model.build_optim()

        self.test_model = Model(
                self.test_dataset, self.config,
                self.global_step, mode='test')

    def train(self, sess, coord):
        sess.run(tf.global_variables_initializer())

        if self.config.pretrain_path is not None:
            self.train_model.restore(sess)

        self.train_dataset.start_feed(sess, coord)
        self.test_dataset.start_feed(sess, coord)

        for idx in trange(self.config.max_step):
            step = self.train_model.run(sess)

            if idx % self.config.log_step == 0:
                step = self.test_model.run(
                        sess, with_update=False, with_summary=True)
                step = self.train_model.run(
                        sess, with_update=True, with_summary=True)

            if idx % self.config.checkpoint_step == 0:
                self.train_model.save(sess, step)

    def test(self):
        pass
