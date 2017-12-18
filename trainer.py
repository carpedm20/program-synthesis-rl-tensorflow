from tqdm import tqdm
import tensorflow as tf
from models import encoder, decoder

from models import Model
from dataset import KarelDataset

class Trainer(object):
    def __init__(self, config, sess, rng=None):
        self.sess = sess
        self.config = config

        self.dataset = KarelDataset(config, rng)

    def train(self):
        inputs, outputs, codes, code_lengths = self.dataset.get_data('train')
        self.model = Model(self.config, inputs, outputs, codes, code_lengths, self.dataset)

        for epoch in range(self.config.epoch):
            self.model.update(self.sess)

    def test(self):
        inputs, outputs, codes = self.karel_dataset.get_data('train')
        pass
