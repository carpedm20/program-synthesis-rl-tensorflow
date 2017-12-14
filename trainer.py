from tqdm import tqdm
import tensorflow as tf
from models import encoder, decoder

from dataset import KarelDataset
from karel.parser import KarelParser

class Trainer(object):
    def __init__(self, config, rng=None):
        self.config = config

        self.karel_dataset = KarelDataset(config, rng)

    def train(self, sess):

        for epoch in range(self.config.epoch):
            for data in tqdm(self.dataset.epoch('train'),
                             total=self.dataset.count('train'),
                             desc="Epoch {:3d}".format(epoch)):
                self.train_epoch(sess)

    def train_epoch(self, sess):
        pass

    def synthesize(self, sess):
        pass

    def get_iterator(self):
        tf_data = tf.data.Dataset.from_tensor_slices((
                self.karel_dataset.inputs['train'],
                self.karel_dataset.outputs['train'],
                self.karel_dataset.codes['train'],
        ))

        if self.config.train:
            batch_size = self.config.batch_size
        else:
            batch_size = 1

        batched_dataset = dataset.batch(batch_size)
        return batched_dataset.make_one_shot_iterator()
