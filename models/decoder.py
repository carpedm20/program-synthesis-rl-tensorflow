import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import Helper, TrainingHelper, BasicDecoder
from tensorflow.contrib.rnn import \
        RNNCell, LSTMCell, MultiRNNCell, OutputProjectionWrapper

from .encoder import linear, int_shape

def decoder(num_examples, codes, code_lengths,
            encoder_out, dataset, config, train_or_test):
    """ codes: [B, L]
        encoder_out: [BxN, 512]
    """
    batch_size = tf.shape(codes)[0]
    batch_times_N = batch_size * num_examples

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
                code_embed, [num_examples, 1, 1], name="tiled_code_embed")
        # [BxN, 256]
        start_code_embed = tf.tile(
                [[0.0]], [batch_times_N, 256], name="start_token")

        # [BxN, L, 768]
        rnn_train_inputs = tf.concat(
                [tiled_encoder_out, tiled_code_embed], -1, name="rnn_input")

        decoder_cell = MultiRNNCell([
                LSTMCell(256),
                NaiveOutputProjectionWrapper(
                    MaxPoolWrapper(LSTMCell(256), num_examples),
                    num_token)], state_is_tuple=True)

        # [BxN, L, 256] -> [B, L, 256] #-> [BxN, L, 256]
        decoder_logits, decoder_argamx = build_rnn(
                train_or_test,
                cell=decoder_cell,
                rnn_train_inputs=rnn_train_inputs,
                start_code_embed=start_code_embed,
                batch_size=batch_times_N,
                target_lengths=code_lengths,
                embedding=embed,
                encoder_out=encoder_out,
                name="decoder_rnn")

        # [BxN, L, 256] -> [B, L, 256]
        decoder_logits = decoder_logits[:batch_size]
        decoder_argamx = decoder_argamx[:batch_size]

    if config.use_syntax:
        syntax_cell = MultiRNNCell([
                LSTMCell(256),
                LSTMCell(256)], state_is_tuple=True)

        # [B, L, 256]
        syntax_out = build_rnn(
                train_or_test,
                cell=syntax_cell,
                rnn_train_inputs=code_embed,
                start_code_embed=start_code_embed,
                batch_size=batch_size,
                target_lengths=code_lengths,
                name="syntax_rnn")

        syntax_logits = linear(max_pool, num_token, "out")

        raise NotImplementedError("TODO")
        decoder_logits = decoder_logits + syntax_logits

    return decoder_logits, decoder_argamx

def build_rnn(train_or_test, cell, rnn_train_inputs, start_code_embed,
              batch_size, target_lengths, embedding, encoder_out, name):

    if train_or_test:
        helper = DefaultZeroInputTrainingHelper(
                rnn_train_inputs, target_lengths, encoder_out, start_code_embed)
    else:
        helper = TestEmbeddingConcatHelper(
                batch_size, embedding, encoder_out, start_code_embed)

    initial_state = cell.zero_state(
            batch_size=batch_size, dtype=tf.float32)

    (decoder_outputs, decoder_samples), final_decoder_state, _ = \
            tf.contrib.seq2seq.dynamic_decode(
                    BasicDecoder(cell, helper, initial_state), scope=name)

    return decoder_outputs, decoder_samples


class DefaultZeroInputTrainingHelper(TrainingHelper):
    def __init__(self, inputs, sequence_length, encoder_out, start_code_embed,
                 time_major=False, name=None):
        super(DefaultZeroInputTrainingHelper, self). \
                __init__(inputs, sequence_length, time_major, name)

        self._start_inputs = tf.concat([
                encoder_out, start_code_embed], -1)

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs)


class TestEmbeddingConcatHelper(Helper):
    def __init__(self, batch_size, embedding, encoder_out, start_code_embed):
        # batch_times_N
        self._batch_size = batch_size
        self._encoder_out = encoder_out
        self._start_code_embed = start_code_embed

        self._embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))
        self._start_inputs = tf.concat([
                self._encoder_out, self._start_code_embed], -1)
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
        #del time, outputs # unused by next_inputs_fn
        finished = tf.equal(sample_ids, self._end_token)
        all_finished = tf.reduce_all(finished)
        sampled_embed = tf.cond(
                all_finished,
                lambda: self._start_code_embed,
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

        # [Bx2, 256]
        max_pool = tf.tile(max_pool, [self._num_examples, 1])

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
