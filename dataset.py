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
    import os
    import argparse
    import numpy as np

    from karel import KarelParser
    from karel import str2bool, makedirs, pprint, beautify

    try:
        from tqdm import trange
    except:
        trange = range

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train_num', type=int, default=1000000)
    arg_parser.add_argument('--test_num', type=int, default=5000)
    arg_parser.add_argument('--val_num', type=int, default=5000)
    arg_parser.add_argument('--data_dir', type=str, default='data')
    arg_parser.add_argument('--max_depth', type=int, default=5)
    arg_parser.add_argument('--mode', type=str, default='all')
    arg_parser.add_argument('--beautify', type=str2bool, default=False)
    arg_parser.add_argument('--world_height', type=int, default=8, help='Height of square grid world')
    arg_parser.add_argument('--world_width', type=int, default=8, help='Width of square grid world')
    args = arg_parser.parse_args()

    # Make directories
    makedirs(args.data_dir)
    datasets = ['train', 'test', 'val']

    # Generate datasets
    parser = KarelParser()

    if args.mode == 'text':
        for name in datasets:
            data_num = getattr(args, "{}_num".format(name))

            text = ""
            text_path = os.path.join(args.data_dir, "{}.txt".format(name))

            for _ in trange(data_num):
                code = parser.random_code(stmt_max_depth=args.max_depth)
                if args.beautify:
                    code = beautify(code)
                text += code  + "\n"

            with open(text_path, 'w') as f:
                f.write(text)
    else:
        for name in datasets:
            data_num = getattr(args, "{}_num".format(name))

            inputs, outputs, codes = [], [], []
            for _ in trange(data_num):
                while True:
                    parser.new_game(world_size=(args.world_width, args.world_height))
                    input = parser.get_state()

                    code = parser.random_code(stmt_max_depth=args.max_depth)

                    try:
                        parser.run(code)
                        output = parser.get_state()
                    except TimeoutError:
                        continue
                    except IndexError:
                        continue

                    inputs.append(input)
                    outputs.append(output)

                    if args.beautify:
                        code = beautify(code)
                    codes.append(code)

                    #pprint(code)
                    break

            npz_path = os.path.join(args.data_dir, name)
            np.savez(npz_path, inputs=inputs, outputs=outputs, codes=codes)
