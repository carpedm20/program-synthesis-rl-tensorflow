from tqdm import trange
import tensorflow as tf
from models import encoder, decoder

from models import Model
from dataset import KarelDataset

class Trainer(object):
    def __init__(self, config, rng=None):
        self.config = config
        self.dataset = KarelDataset(config, rng)

        inputs, outputs, codes, code_lengths = \
                self.dataset.get_data('train' if config.train else 'test')

        self.model = Model(self.config, inputs, outputs, codes, code_lengths, self.dataset)
        self.initializer = tf.global_variables_initializer()

    def train(self, sess):
        for epoch in trange(self.config.epoch):
            self.model.update(sess)

    def test(self):
        pass
