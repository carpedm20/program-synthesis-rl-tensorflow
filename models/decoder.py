import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import Helper, TrainingHelper, BasicDecoder
from tensorflow.contrib.rnn import \
        RNNCell, LSTMCell, MultiRNNCell, OutputProjectionWrapper

from .encoder import linear, int_shape

class Decoder(object):
    """ B: batch_size
        N: num_example
        L: max length of code in a batch
    """
    def __init__(self, config, codes, encoder_out, code_lengths, dataset):
        """ codes: [B, L]
            encoder_out: [BxN, 512]
        """
        batch_size = tf.shape(codes)[0]
        num_token = dataset.num_token

        with tf.variable_scope("decoder"):
            # [BxN, L, 512]
            tiled_encoder_out = tf.tile(
                    tf.expand_dims(encoder_out, 1),
                    [1, tf.shape(codes)[1], 1],
                    name="tiled_encoder_out")

            embed = tf.get_variable(
                    'embedding', [dataset.num_token, 256], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.5))

            # [B, L, 256]
            code_embed = tf.nn.embedding_lookup(embed, codes)
            # [BxN, L, 256]
            tiled_code_embed = tf.tile(
                    code_embed, [2, 1, 1], name="tiled_code_embed")

            # [BxN, L, 768]
            rnn_input = tf.concat(
                    [tiled_code_embed, tiled_encoder_out], -1, name="rnn_input")

            shape = tf.shape(rnn_input)
            batch_times_N, L, input_dim = shape[0], shape[1], shape[2]

            decoder_cell = MultiRNNCell([
                    LSTMCell(256),
                    NaiveOutputProjectionWrapper(
                        MaxPoolWrapper(LSTMCell(256), config.num_examples),
                        num_token)], state_is_tuple=True)

            # [BxN, L, 256] -> [B, L, 256]
            decoder_logits = build_rnn(
                    config,
                    cell=decoder_cell,
                    inputs=tiled_code_embed,
                    batch_size=batch_times_N,
                    name="decoder_rnn",
                    target_lengths=code_lengths,
                    embedding=embed,
                    encoder_out=encoder_out,
                    output_dim=num_token)

        if config.use_syntax:
            syntax_cell = MultiRNNCell([
                    LSTMCell(256),
                    LSTMCell(256)], state_is_tuple=True)

            # [B, L, 256]
            syntax_out = build_rnn(
                    config,
                    cell=syntax_cell,
                    inputs=code_embed,
                    batch_size=batch_size,
                    name="syntax_rnn",
                    target_lengths=code_lengths)

            syntax_logits = linear(max_pool, num_token, "out")

            raise NotImplementedError("TODO")
            decoder_logits = decoder_logits + syntax_logits

        self.logits = decoder_logits

def build_rnn(config, cell, inputs, batch_size, name,
              target_lengths=None, embedding=None, encoder_out=None, output_dim=None):
    if config.train:
        helper = TrainingHelper(inputs, target_lengths)
    else:
        helper = TestEmbeddingConcatHelper(batch_size, embedding, encoder_out, output_dim=output_dim)

    initial_state = cell.zero_state(
            batch_size=batch_size, dtype=tf.float32)

    (decoder_outputs, _), final_decoder_state, _ = \
            tf.contrib.seq2seq.dynamic_decode(
                    BasicDecoder(cell, helper, initial_state), scope=name)

    return decoder_outputs


class TestEmbeddingConcatHelper(Helper):
    def __init__(self, batch_size, embedding, encoder_out, output_dim):
        self._batch_size = batch_size
        self._encoder_out = encoder_out
        self._output_dim = output_dim

        self._embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))
        self._start_inputs = \
                tf.tile([[0.0]], [batch_size, output_dim], name="_start_token")
        self._end_token = 0

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        #del time, state # unused by sample_fn
        sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        #del time, outputs    # unused by next_inputs_fn
        finished = tf.equal(sample_ids, self._end_token)
        all_finished = tf.reduce_all(finished)
        sampled_embed = tf.cond(
                all_finished,
                lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))
        next_inputs = tf.concat([self._encoder_out, sampled_embed], -1)
        return (finished, next_inputs, state)


class MaxPoolWrapper(RNNCell):
    def __init__(self, cell, num_examples, reuse=None):
        super(MaxPoolWrapper, self).__init__(_reuse=reuse)
        self._cell = cell
        self._num_examples = num_examples

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        B_times_N, cell_dim = int_shape(output)

        # [BxN, 256] -> [B, N, 256]
        decoder_out = tf.reshape(
                output, [-1, self._num_examples, cell_dim])

        # [B, 256]
        max_pool = tf.reduce_max(decoder_out, 1)
        return max_pool, res_state


class NaiveOutputProjectionWrapper(OutputProjectionWrapper):
    def __init__(self, cell, output_size, activation=None, reuse=None):
        try:
            super(NaiveOutputProjectionWrapper, self). \
                    __init__(cell, output_size, activation, reuse)
        except TypeError:
            pass

        self._cell = cell
        self._output_size = output_size
        self._activation = activation
        self._linear = None
