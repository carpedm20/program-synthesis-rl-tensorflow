import os
import numpy as np
import tensorflow as tf
from collections import namedtuple

from karel import KarelParser
from karel import str2bool, makedirs, pprint, beautify, TimeoutError

from utils import get_rng

Data = namedtuple('Data', 'input, output, code')

class Dataset(object):
    tokens = []
    idx_to_token = {}

    def __init__(self, config, rng=None, shuffle=False):
        self.config = config
        self.rng = get_rng(rng)

        self.inputs, self.outputs, self.codes, self.code_lengths = {}, {}, {}, {}
        self._inputs, self._outputs, self._codes, self._code_lengths = {}, {}, {}, {}

        self.data_names = ['train', 'test', 'val']
        self.data_paths = {
                key: os.path.join(config.data_dir, '{}.{}'.format(key, config.data_ext)) \
                        for key in self.data_names
        }

        self.load_data()
        if shuffle:
            self.shuffle()

        for name in self.data_names:
            inputs, outputs, codes, code_lengths = self.build_tf_data(name)

            self.inputs[name] = inputs
            self.outputs[name] = outputs
            self.codes[name] = codes
            self.code_lengths[name] = code_lengths

    def build_tf_data(self, name):
        if self.config.train:
            batch_size = self.config.batch_size
        else:
            batch_size = 1

        # inputs, outputs
        in_out = tf.data.Dataset.from_tensor_slices((
                self._inputs[name], self._outputs[name], self._code_lengths[name]
        ))
        batched_in_out = in_out.batch(batch_size)

        # codes
        code = tf.data.Dataset.from_generator(lambda: self._codes[name], tf.int32)
        batched_code = code.padded_batch(batch_size, padded_shapes=[None])

        batched_data = tf.data.Dataset.zip((batched_in_out, batched_code))
        (inputs, outputs, code_lengths), codes = batched_data.make_one_shot_iterator().get_next()

        inputs = tf.cast(inputs, tf.float32)
        outputs = tf.cast(outputs, tf.float32)
        code_lengths = tf.cast(code_lengths, tf.int32)

        return inputs, outputs, codes, code_lengths

    def get_data(self, name):
        return self.inputs[name], self.outputs[name], self.codes[name], self.code_lengths[name]

    def count(self, name):
        return len(self._inputs[name])

    def shuffle(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    @property
    def token_num(self):
        return len(self.tokens)

class KarelDataset(Dataset):
    def __init__(self, config, *args, **kwargs):
        super(KarelDataset, self).__init__(config, *args, **kwargs)

        self.parser = KarelParser()

        self.tokens = ['END'] + self.parser.tokens
        self.idx_to_token = { idx: token for idx, token in enumerate(self.tokens) }

    def load_data(self):
        self.data = {}
        for name in self.data_names:
            data = np.load(self.data_paths[name])
            self._inputs[name] = data['inputs']
            self._outputs[name] = data['outputs']
            self._codes[name] = data['codes']
            self._code_lengths[name] = data['code_lengths']

    def shuffle(self):
        for name in self.data_names:
            self.rng.shuffle(self._inputs[name])
            self.rng.shuffle(self._outputs[name])
            self.rng.shuffle(self._codes[name])


if __name__ == '__main__':
    import os
    import argparse
    import numpy as np

    try:
        from tqdm import trange
    except:
        trange = range
    
    from config import get_config
    config, _ = get_config()

    # Make directories
    makedirs(config.data_dir)
    datasets = ['train', 'test', 'val']

    # Generate datasets
    parser = KarelParser()

    if config.mode == 'text':
        for name in datasets:
            data_num = getattr(config, "num_{}".format(name))

            text = ""
            text_path = os.path.join(config.data_dir, "{}.txt".format(name))

            for _ in trange(data_num):
                code = parser.random_code(stmt_max_depth=config.max_depth)
                if config.beautify:
                    code = beautify(code)
                text += code  + "\n"

            with open(text_path, 'w') as f:
                f.write(text)
    else:
        for name in datasets:
            data_num = getattr(config, "num_{}".format(name))

            inputs, outputs, codes, code_lengths = [], [], [], []
            for _ in trange(data_num):
                while True:
                    input_examples, output_examples = [], []

                    code = parser.random_code(stmt_max_depth=config.max_depth)
                    #pprint(code)

                    num_code_error, resample_code = 0, False
                    while len(input_examples) < config.num_examples:
                        if num_code_error > 5:
                            resample_code = True
                            break

                        parser.new_game(world_size=(config.world_width, config.world_height))
                        input = parser.get_state()

                        try:
                            parser.run(code)
                            output = parser.get_state()
                        except TimeoutError:
                            num_code_error += 1
                            continue
                        except IndexError:
                            num_code_error += 1
                            continue

                        input_examples.append(input)
                        output_examples.append(output)

                    if resample_code:
                        continue

                    inputs.append(input_examples)
                    outputs.append(output_examples)

                    token_idxes = parser.lex_to_idx(code)

                    # Add END tokens for seq2seq prediction
                    token_idxes = np.array(token_idxes, dtype=np.uint8) + 1
                    token_idxes_with_end = np.append(token_idxes, [0])

                    codes.append(token_idxes_with_end)
                    code_lengths.append(len(token_idxes_with_end))
                    break

            npz_path = os.path.join(config.data_dir, name)
            np.savez(npz_path,
                     inputs=inputs,
                     outputs=outputs,
                     codes=codes,
                     code_lengths=code_lengths)
