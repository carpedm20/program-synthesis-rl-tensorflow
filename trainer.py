from models import encoder, decoder

from dataset import KarelDataset

class Trainer(object):
    def __init__(self, config, rng=None):
        self.config = config

        self.dataset = KarelDataset(config, rng)

    def train(self):
        for epoch in range(self.config.epoch):
            for data in self.dataset.epoch('train'):
                self.train_epoch()

    def train_epoch(self):
        pass

    def synthesize(self):
        pass
