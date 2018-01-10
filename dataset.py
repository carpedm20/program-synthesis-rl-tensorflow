import os
import threading
import traceback
import numpy as np
import tensorflow as tf
from collections import namedtuple, defaultdict

from utils import get_rng
from karel import KarelForSynthesisParser, InvalidActionError
from karel import str2bool, makedirs, pprint, beautify, TimeoutError


Data = namedtuple('Data', 'input, output, code')

def try_beautify(x):
    try:
        x = beautify(x)
    except:
        pass
    return x

class Dataset(threading.Thread):
    tokens = []
    idx_to_token = {}

    def __init__(self, mode, config, rng=None, load=True, shuffle=True):
        super(Dataset, self).__init__()

        self.sess = None
        self.coord = None

        self.idx = 0
        self.mode = mode
        self.config = config
        self.rng = get_rng(rng)

        self.data_path = os.path.join(
                config.data_dir, '{}.{}'.format(mode, config.data_ext))

        self.with_input_string = False

        if load:
            self.load_data()
            self.build_tf_queue()
            #self.build_tf_data()

        if shuffle:
            self.shuffle()

    def get_data(self):
        data = {
                'inputs': self.inputs,
                'outputs': self.outputs,
                'codes': self.codes,
                'code_lengths': self.code_lengths,
        }
        if self.with_input_string:
            data.update({
                'input_strings': self.input_strings,
                'output_strings': self.output_strings,
            })
        return data

    def build_tf_queue(self):
        self.inputs_placeholder = tf.placeholder(
                tf.float32, list(self._inputs[0].shape),
                name="self.inputs_placeholder")
        self.outputs_placeholder = tf.placeholder(
                tf.float32, list(self._outputs[0].shape),
                name="outputs_placeholder")
        self.codes_placeholder = tf.placeholder(
                tf.int32, [None],
                name="codes_placeholder")
        self.code_lengths_placeholder = tf.placeholder(
                tf.int32, [],
                name="code_lengths_placeholder")

        self.placeholders = [
                self.inputs_placeholder, self.outputs_placeholder,
                self.codes_placeholder, self.code_lengths_placeholder,
        ]

        if self.with_input_string:
            self.input_strings_placeholder = tf.placeholder(
                    tf.string, list(self._input_strings[0].shape),
                    name="input_strings_placeholder")
            self.output_strings_placeholder = tf.placeholder(
                    tf.string, list(self._output_strings[0].shape),
                    name="output_strings_placeholder")

            self.placeholders.extend([
                    self.input_strings_placeholder,
                    self.output_strings_placeholder
            ])

        self.queue = tf.PaddingFIFOQueue(
                self.config.batch_size,
                [op.dtype for op in self.placeholders],
                [op.shape for op in self.placeholders],
                name='input_queue')
        self.enqueue_op = self.queue.enqueue(self.placeholders)

        if self.with_input_string:
            self.inputs, self.outputs, \
                    self.codes, self.code_lengths, \
                    self.input_strings, self.output_strings \
                    = self.queue.dequeue_many(self.config.batch_size)
        else:
            self.inputs, self.outputs, \
                    self.codes, self.code_lengths \
                    = self.queue.dequeue_many(self.config.batch_size)

        #self.inputs.set_shape(self.inputs_placeholder.shape)
        #self.outputs.set_shape(self.outputs_placeholder.shape)
        #self.codes.set_shape(self.codes_placeholder.shape)
        #self.code_lengths.set_shape(self.code_lengths_placeholder.shape)

        #if self.with_input_string:
        #    self.input_strings.set_shape(
        #            self.input_strings_placeholder.shape)
        #    self.output_strings.set_shape(
        #            self.output_strings_placeholder.shape)

    def enqueue(self):
        idx = self.data_idx[self.idx]

        feed_dict = {
                self.inputs_placeholder: self._inputs[idx],
                self.outputs_placeholder: self._outputs[idx],
                self.codes_placeholder: self._codes[idx],
                self.code_lengths_placeholder: self._code_lengths[idx],
        }
        if self.with_input_string:
            feed_dict.update({
                self.input_strings_placeholder: self._input_strings[idx],
                self.output_strings_placeholder: self._output_strings[idx],
            })

        self.sess.run(self.enqueue_op, feed_dict)

        self.idx += 1
        if self.idx >= len(self._inputs):
            self.idx = 0
            self.shuffle()

    def start_feed(self, sess, coord):
        self.sess = sess
        self.coord = coord
        self.start()

    def run(self):
        try:
            while not self.coord.should_stop():
                self.enqueue()
        except Exception as e:
            traceback.print_exc()
            self.coord.request_stop(e)

    def count(self):
        return len(self._inputs)

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

    def run_with_example(self, batch_code, batch_example, with_test=True, **kwargs):
        batch_output = []
        timeout_count, syntax_error_count, be_able_to_run_count, invalid_move_count = \
                np.zeros([4] + list(batch_example.shape[:2]), dtype=np.float32)

        for idx, (code, examples) in enumerate(zip(batch_code, batch_example)):
            outputs = []
            tokens = [token.decode("utf-8").rstrip('\x00') for token in code.split()]

            try:
                code = " ".join([token for token in tokens[:tokens.index('END')]])

                for jdx, state in enumerate(examples):
                    try:
                        self.parser.new_game(state=state)
                        self.parser.run(code, **kwargs)
                        output = self.parser.draw_for_tensorboard()
                        be_able_to_run_count[idx][jdx] = 1
                    except TimeoutError:
                        output = 'time'
                        timeout_count[idx][jdx] = 1
                    except TypeError:
                        output = 'type'
                        syntax_error_count[idx][jdx] = 1
                    except InvalidActionError:
                        output = 'invalid'
                        invalid_move_count[idx][jdx] = 1
                    outputs.append(output)
            except ValueError:
                outputs = ['no_end'] * len(examples)

            batch_output.append(outputs)

        if with_test:
            return np.array(batch_output), timeout_count, \
                    syntax_error_count, be_able_to_run_count, invalid_move_count
        else:
            return np.array(batch_output)

    def build_tf_data(self):
        if self.config.train:
            batch_size = self.config.batch_size
        else:
            batch_size = 1

        # inputs, outputs
        data = [
                self._inputs, self._outputs, self._code_lengths
        ]
        if self.with_input_string:
            data.extend([self._input_strings, self._output_strings])

        in_out = tf.data.Dataset.from_tensor_slices(tuple(data)).repeat()
        batched_in_out = in_out.batch(batch_size)

        # codes
        code = tf.data.Dataset.from_generator(lambda: self._codes, tf.int32).repeat()
        batched_code = code.padded_batch(batch_size, padded_shapes=[None])

        batched_data = tf.data.Dataset.zip((batched_in_out, batched_code))
        iterator = batched_data.make_initializable_iterator()

        if self.with_input_string:
            (inputs, outputs, code_lengths, \
                    input_strings, output_strings), codes = iterator.get_next()

            input_strings = tf.cast(input_strings, tf.string)
            output_strings = tf.cast(output_strings, tf.string)
        else:
            (inputs, outputs, code_lengths), codes = iterator.get_next()

        inputs = tf.cast(inputs, tf.float32)
        outputs = tf.cast(outputs, tf.float32)
        code_lengths = tf.cast(code_lengths, tf.int32)

        self.inputs = inputs
        self.outputs = outputs
        self.codes = codes
        self.code_lengths = code_lengths
        self.iterator = iterator

        if self.with_input_string:
            self.input_strings = input_strings
            self.output_strings = output_strings


class KarelDataset(Dataset):
    def __init__(self, mode, config, *args, **kwargs):
        super(KarelDataset, self).__init__(mode, config, *args, **kwargs)

        self.parser = KarelForSynthesisParser(
                rng=self.rng,
                min_int=config.min_int, max_int=config.max_int,
                max_func_call=config.max_func_call, debug=config.debug)

        self.tokens = self.parser.tokens_details
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
        data = np.load(self.data_path)

        self._inputs = data['inputs']
        self._outputs = data['outputs']
        self._codes = data['codes']
        self._code_lengths = data['code_lengths']

        if 'input_strings' in data:
            self.with_input_string = True
            self._input_strings = data['input_strings']
            self._output_strings = data['output_strings']

        self.data_idx = list(range(len(self._inputs)))

    def shuffle(self):
        self.rng.shuffle(self.data_idx)


def generate_data(mode, config):
    dataset = KarelDataset(mode, config, shuffle=False, load=False)
    parser = dataset.parser

    # Make directories
    makedirs(config.data_dir)

    # Generate datasets
    file_loc = [0]

    if MS_DATA_DIR:
        lines = open(os.path.join(MS_DATA_DIR,
                        "{}.prog".format(mode))).readlines()

        def generate():
            try:
                parser.flush_hit_info()
                code = lines[file_loc[0]]
                file_loc[0] += 1
            except IndexError:
                return False
            return code
    else:
        def generate():
            parser.flush_hit_info()
            code = parser.random_code(
                    stmt_max_depth=config.max_depth,
                    min_move=config.min_move,
                    create_hit_info=True)
            return code

    if config.mode == 'text':
        data_num = getattr(config, "num_{}".format(mode))

        text = ""
        text_path = os.path.join(config.data_dir, "{}.txt".format(mode))

        for _ in trange(data_num, desc=mode):
            code = generate()
            if config.beautify:
                code = beautify(code)
            text += code  + "\n"

        with open(text_path, 'w') as f:
            f.write(text)
    else:
        data_num = getattr(config, "num_{}".format(mode))

        inputs, outputs, codes, code_lengths = [], [], [], []
        input_strings, output_strings = [], []

        terminal = False
        for _ in trange(data_num, desc=mode):
            if terminal:
                break

            while True:
                input_examples, output_examples = [], []
                input_string_examples, output_string_examples = [], []

                code = generate()
                if not code:
                    terminal = True
                    break

                num_io_error, num_timeout_error, resample_code = 0, 0, False
                while len(input_examples) < config.num_spec + config.num_heldout:
                    if num_io_error > 100 or num_timeout_error > 100:
                        resample_code = True
                        #print(beautify(code), num_io_error, num_timeout_error)
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
                        if MS_DATA_DIR in [None, '']:
                            num_timeout_error += 1
                            if DEBUG: print("T", beautify(code))
                            continue
                    except IndexError:
                        num_io_error += 1
                        if DEBUG: print("XX", beautify(code))
                        continue
                    except InvalidActionError:
                        num_timeout_error += 1
                        if DEBUG: print("Game error", beautify(code))
                        continue

                    # input/output pair should be different
                    if np.array_equal(input_state, output_state):
                        num_io_error += 1
                        if DEBUG: print("I/O", beautify(code))
                        continue

                    input_examples.append(input_state)
                    input_string_examples.append(input_string)
                    output_examples.append(output_state)
                    output_string_examples.append(output_string)

                # if there is at least one contionals
                if parser.hit_info is not None:
                    if len(parser.hit_info) > 0:
                        # if there are contionals not hitted
                        if max(parser.hit_info.values()) > 0:
                            continue

                if resample_code:
                    continue

                if DEBUG:
                    print(beautify(code))
                    print(input_string)
                    print(output_string)

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

        npz_path = os.path.join(config.data_dir, mode)
        np.savez(npz_path,
                    inputs=inputs,
                    outputs=outputs,
                    input_strings=input_strings,
                    output_strings=output_strings,
                    codes=codes,
                    code_lengths=code_lengths)

if __name__ == '__main__':
    import os
    import argparse
    import numpy as np

    DEBUG = False
    MS_DATA_DIR = 'ms_karel'
    #MS_DATA_DIR = ''

    try:
        from tqdm import trange
    except:
        trange = lambda x, desc: range(x)
    
    from config import get_config
    config, _ = get_config()

    for mode in ['train', 'test', 'val']:
        generate_data(mode, config)
