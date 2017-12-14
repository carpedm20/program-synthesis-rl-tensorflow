import os
import numpy as np
from collections import namedtuple

from utils import get_rng

Data = namedtuple('Data', 'input, output, code')

class Dataset(object):
    def __init__(self, config, rng=None, shuffle=False):
        self.config = config
        self.rng = get_rng(rng)

        self.inputs = {}
        self.outputs = {}
        self.codes = {}

        self.data_names = ['train', 'test', 'val']
        self.data_paths = {
                key: os.path.join(config.data_dir, '{}.{}'.format(key, config.data_ext)) \
                        for key in self.data_names
        }

        self.load_data()
        if shuffle:
            self.shuffle()

    def epoch(self, name):
        inputs, outputs, codes = \
                self.inputs[name], self.outputs[name], self.codes[name]

        for input_, output, code in zip(inputs, outputs, codes):
            yield Data(input_, output, code)

    def count(self, name):
        return len(self.inputs[name])

    def shuffle(self):
        pass

    def load_data(self):
        pass

class KarelDataset(Dataset):
    def __init__(self, config, *args, **kwargs):
        super(KarelDataset, self).__init__(config, *args, **kwargs)

    def load_data(self):
        self.data = {}
        for name in self.data_names:
            data = np.load(self.data_paths[name])
            self.inputs[name] = data['inputs']
            self.outputs[name] = data['outputs']
            self.codes[name] = data['codes']

    def shuffle(self):
        for name in self.data_names:
            self.rng.shuffle(self.inputs[name])
            self.rng.shuffle(self.outputs[name])
            self.rng.shuffle(self.codes[name])


if __name__ == '__main__':
    from config import get_config
    config, _ = get_config()

    dataset = KarelDataset(config)
    for data in dataset.epoch('train'):
        print(data.code)
