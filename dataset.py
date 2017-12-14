import os
import numpy as np
import tensorflow as tf
from collections import namedtuple

from utils import get_rng

Data = namedtuple('Data', 'input, output, code')

class Dataset(object):
    def __init__(self, config, rng=None, shuffle=False):
        self.config = config
        self.rng = get_rng(rng)

        self.inputs, self.outputs, self.codes = {}, {}, {}
        self._inputs, self._outputs, self._codes = {}, {}, {}

        self.data_names = ['train', 'test', 'val']
        self.data_paths = {
                key: os.path.join(config.data_dir, '{}.{}'.format(key, config.data_ext)) \
                        for key in self.data_names
        }

        self.load_data()
        if shuffle:
            self.shuffle()

        for name in self.data_names:
            inputs, outputs, codes = self.build_tf_data(name)

            self.inputs[name] = inputs
            self.outputs[name] = outputs
            self.codes[name] = codes

    def build_tf_data(self, name):
        tf_data = tf.data.Dataset.from_tensor_slices((
                self._inputs[name], self._outputs[name], self._codes[name],
        ))

        if self.config.train:
            batch_size = self.config.batch_size
        else:
            batch_size = 1

        batched_data = tf_data.batch(batch_size)
        iterator = batched_data.make_one_shot_iterator()
        inputs, outputs, codes = iterator.get_next()

        return inputs, outputs, codes

    def get_data(self, name):
        return self.inputs[name], self.outputs[name], self.codes[name]

    def count(self, name):
        return len(self._inputs[name])

    def shuffle(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

class KarelDataset(Dataset):
    def __init__(self, config, *args, **kwargs):
        super(KarelDataset, self).__init__(config, *args, **kwargs)

    def load_data(self):
        self.data = {}
        for name in self.data_names:
            data = np.load(self.data_paths[name])
            self._inputs[name] = data['inputs']
            self._outputs[name] = data['outputs']
            self._codes[name] = data['codes']

    def shuffle(self):
        for name in self.data_names:
            self.rng.shuffle(self._inputs[name])
            self.rng.shuffle(self._outputs[name])
            self.rng.shuffle(self._codes[name])


if __name__ == '__main__':
    from config import get_config
    config, _ = get_config()

    dataset = KarelDataset(config)
    for data in dataset.epoch('train'):
        print(data.code)
