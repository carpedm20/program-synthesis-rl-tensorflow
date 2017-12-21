import os
import numpy as np
import tensorflow as tf
from collections import namedtuple

from utils import get_rng
from karel import KarelForSynthesisParser
from karel import str2bool, makedirs, pprint, beautify, TimeoutError

Data = namedtuple('Data', 'input, output, code')

def try_beautify(x):
    try:
        x = beautify(x)
    except:
        pass
    return x

class Dataset(object):
    tokens = []
    idx_to_token = {}

    def __init__(self, config, rng=None, load=True, shuffle=False):
        self.config = config
        self.rng = get_rng(rng)

        self.inputs, self.outputs, self.codes, self.code_lengths = {}, {}, {}, {}
        self.input_strings, self.output_strings = {}, {}
        self.with_input_string = False

        self.iterator = {}
        self._inputs, self._outputs, self._codes, self._code_lengths = {}, {}, {}, {}
        self._input_strings, self._output_strings = {}, {}

        self.data_names = ['train', 'test', 'val']
        self.data_paths = {
                key: os.path.join(config.data_dir, '{}.{}'.format(key, config.data_ext)) \
                        for key in self.data_names
        }

        if load:
            self.load_data()
            for name in self.data_names:
                self.build_tf_data(name)
        if shuffle:
            self.shuffle()

    def build_tf_data(self, name):
        if self.config.train:
            batch_size = self.config.batch_size
        else:
            batch_size = 1

        # inputs, outputs
        data = [
                self._inputs[name], self._outputs[name], self._code_lengths[name]
        ]
        if self.with_input_string:
            data.extend([self._input_strings[name], self._output_strings[name]])

        in_out = tf.data.Dataset.from_tensor_slices(tuple(data)).repeat()
        batched_in_out = in_out.batch(batch_size)

        # codes
        code = tf.data.Dataset.from_generator(lambda: self._codes[name], tf.int32).repeat()
        batched_code = code.padded_batch(batch_size, padded_shapes=[None])

        batched_data = tf.data.Dataset.zip((batched_in_out, batched_code))
        iterator = batched_data.make_initializable_iterator()

        if self.with_input_string:
            (inputs, outputs, code_lengths, input_strings, output_strings), codes = iterator.get_next()

            input_strings = tf.cast(input_strings, tf.string)
            output_strings = tf.cast(output_strings, tf.string)
        else:
            (inputs, outputs, code_lengths), codes = iterator.get_next()

        inputs = tf.cast(inputs, tf.float32)
        outputs = tf.cast(outputs, tf.float32)
        code_lengths = tf.cast(code_lengths, tf.int32)

        self.inputs[name] = inputs
        self.outputs[name] = outputs
        self.codes[name] = codes
        self.code_lengths[name] = code_lengths
        self.iterator[name] = iterator

        if self.with_input_string:
            self.input_strings[name] = input_strings
            self.output_strings[name] = output_strings

    def get_data(self, name):
        data = {
                'inputs': self.inputs[name],
                'outputs': self.outputs[name],
                'codes': self.codes[name],
                'code_lengths': self.code_lengths[name],
                'iterator': self.iterator[name]
        }
        if self.with_input_string:
            data.update({
                'input_strings': self.input_strings[name],
                'output_strings': self.output_strings[name],
            })
        return data

    def count(self, name):
        return len(self._inputs[name])

    def shuffle(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    @property
    def num_token(self):
        return len(self.tokens)

    def _idx_to_text(self, idxes, beautify):
        code = " ".join(self.token_to_text[
                self.parser.idx_to_token_details[idx]] for idx in idxes).replace("\\", "")
        if beautify:
            code = try_beautify(code)
        return code

    def idx_to_text(self, idxes, markdown=False, beautify=False):
        if hasattr(idxes[0], '__len__'):
            if markdown:
                strings = ["\t{}".format(self._idx_to_text(idxes, beautify) \
                                             .replace('\n', '\n\t')) for idxes in idxes]
            else:
                strings = [self._idx_to_text(idxes, beautify) for idxes in idxes]
        else:
            strings = self._idx_to_text(idxes, beautify)
        return np.array(strings)

    def run_and_test(self, batch_code, batch_example, **kwargs):
        batch_output = []
        for code, examples in zip(batch_code, batch_example):
            outputs = []
            tokens = [token.decode("utf-8") for token in code.split()]

            try:
                code = " ".join([token for token in tokens[:tokens.index('END')]])

                for state in examples:
                    try:
                        self.parser.new_game(state=state)
                        self.parser.run(code, **kwargs)
                        output = self.parser.draw_for_tensorboard()
                    except TimeoutError:
                        output = 'time'
                    except TypeError:
                        output = 'type'
                    except ValueError:
                        output = 'value'
                    outputs.append(output)
            except ValueError:
                outputs = ['no_end'] * len(examples)

            batch_output.append(outputs)
        return np.array(batch_output)


class KarelDataset(Dataset):
    def __init__(self, config, *args, **kwargs):
        super(KarelDataset, self).__init__(config, *args, **kwargs)

        self.parser = KarelForSynthesisParser(
                rng=self.rng, max_func_call=config.max_func_call, debug=config.debug)

        self.tokens = ['END'] + self.parser.tokens_details
        self.token_to_text = { 'END': 'END' }

        for token in self.tokens:
            if token in ['END']:
                continue
            elif token.startswith('INT'):
                self.token_to_text[token] = token.replace('INT', self.parser.INT_PREFIX)
                continue

            item = getattr(self.parser, 't_{}'.format(token))
            if callable(item):
                self.token_to_text[token] = token
            else:
                self.token_to_text[token] = item

    def load_data(self):
        self.data = {}
        for name in self.data_names:
            data = np.load(self.data_paths[name])
            self._inputs[name] = data['inputs']
            self._outputs[name] = data['outputs']
            self._codes[name] = data['codes']
            self._code_lengths[name] = data['code_lengths']

            if 'input_strings' in data:
                self.with_input_string = True
                self._input_strings[name] = data['input_strings']
                self._output_strings[name] = data['output_strings']

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
        trange = lambda x, desc: range(x)
    
    from config import get_config
    config, _ = get_config()

    dataset = KarelDataset(config, load=False)
    parser = dataset.parser

    # Make directories
    makedirs(config.data_dir)
    datasets = ['train', 'test', 'val']

    # Generate datasets

    def generate():
        parser.flush_hit_info()
        code = parser.random_code(
                stmt_max_depth=config.max_depth,
                min_move=config.min_move,
                create_hit_info=True)
        return code

    if config.mode == 'text':
        for name in datasets:
            data_num = getattr(config, "num_{}".format(name))

            text = ""
            text_path = os.path.join(config.data_dir, "{}.txt".format(name))

            for _ in trange(data_num, desc=name):
                code = generate()
                if config.beautify:
                    code = beautify(code)
                text += code  + "\n"

            with open(text_path, 'w') as f:
                f.write(text)
    else:
        for name in datasets:
            data_num = getattr(config, "num_{}".format(name))

            inputs, outputs, codes, code_lengths = [], [], [], []
            input_strings, output_strings = [], []

            for _ in trange(data_num, desc=name):
                while True:
                    input_examples, output_examples = [], []
                    input_string_examples, output_string_examples = [], []

                    code = generate()
                    #pprint(code)

                    num_code_error, resample_code = 0, False
                    while len(input_examples) < config.num_spec + config.num_heldout:
                        if num_code_error > 5:
                            resample_code = True
                            break

                        parser.new_game(
                                world_size=(config.world_width, config.world_height),
                                wall_ratio=config.wall_ratio, marker_ratio=config.marker_ratio)
                        input_string = parser.draw_for_tensorboard()
                        input_state = parser.get_state()

                        try:
                            parser.run(code, debug=config.parser_debug)
                            output_state = parser.get_state()
                            output_string = parser.draw_for_tensorboard()
                        except TimeoutError:
                            num_code_error += 1
                            continue
                        except IndexError:
                            num_code_error += 1
                            continue

                        # input/output pair should be different
                        if np.array_equal(input_state, output_state):
                            num_code_error += 1
                            continue

                        input_examples.append(input_state)
                        input_string_examples.append(input_string)
                        output_examples.append(output_state)
                        output_string_examples.append(output_string)

                    # if there is at least one contionals
                    if len(parser.hit_info) > 0:
                        # if there are contionals not hitted
                        if max(parser.hit_info.values()) > 0:
                            continue

                    if resample_code:
                        continue

                    inputs.append(input_examples)
                    outputs.append(output_examples)

                    input_strings.append(input_string_examples)
                    output_strings.append(output_string_examples)

                    token_idxes = parser.lex_to_idx(code, details=True)

                    # Add END tokens for seq2seq prediction
                    token_idxes = np.array(token_idxes, dtype=np.uint8)
                    token_idxes_with_end = np.append(
                            token_idxes, parser.token_to_idx_details['END'])

                    codes.append(token_idxes_with_end)
                    code_lengths.append(len(token_idxes_with_end))
                    break

            npz_path = os.path.join(config.data_dir, name)
            np.savez(npz_path,
                     inputs=inputs,
                     outputs=outputs,
                     input_strings=input_strings,
                     output_strings=output_strings,
                     codes=codes,
                     code_lengths=code_lengths)
