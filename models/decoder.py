import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import RNNCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.seq2seq import TrainingHelper, GreedyEmbeddingHelper, BasicDecoder

from .encoder import linear

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
            tiled_cnn_out = tf.tile(
                    tf.expand_dims(encoder_out, 1),
                    [1, tf.shape(codes)[1], 1],
                    name="tiled_cnn_out")

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
                    [tiled_code_embed, tiled_cnn_out], -1, name="rnn_input")

            shape = tf.shape(rnn_input)
            batch_times_N, L, input_dim = shape[0], shape[1], shape[2]

            decoder_cell = MultiRNNCell([
                    LSTMCell(256),
                    LSTMCell(256),
                    MaxPoolWrapper(config.num_examples),
                    OutputProjectionWrapper(num_token)], state_is_tuple=True)

            # [BxN, L, 256]
            decoder_logits = build_rnn(
                    config,
                    cell=decoder_cell,
                    inputs=tiled_code_embed,
                    batch_size=batch_times_N,
                    name="decoder_rnn",
                    target_lengths=code_lengths)

            # [B, N, L, 256]
            decoder_out = tf.reshape(
                    tiled_decoder_out,
                    [batch_size, config.num_examples, L, 256])

            # [B, L, 256]
            max_pool = tf.reduce_max(decoder_out, 1)

            decoder_logits = linear(max_pool, num_token, "out")

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

def build_rnn(config, cell, inputs, batch_size, name, target_lengths=None):
    if config.train:
        helper = TrainingHelper(inputs, target_lengths)
    else:
        helper = GreedyEmbeddingHelper(inputs, end_token=0)

    initial_state = cell.zero_state(
            batch_size=batch_size, dtype=tf.float32)

    (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
            BasicDecoder(cell, helper, initial_state), scope=name)

    return decoder_outputs


class TestHelper(Helper):
    def __init__(self, batch_size, output_dim):
        with tf.name_scope('TacoTestHelper'):
            self._batch_size = batch_size
            self._output_dim = output_dim
            self._end_token = tf.tile([0.0], [output_dim * r])

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
        return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])    # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        '''Stop on EOS. Otherwise, pass the last output as the next input and pass through state.'''
        with tf.name_scope('TacoTestHelper'):
            finished = tf.reduce_all(tf.equal(outputs, self._end_token), axis=1)
            # Feed last output frame as next input. outputs is [N, output_dim * r]
            next_inputs = outputs[:, -self._output_dim:]
            return (finished, next_inputs, state)


class MaxPoolWrapper(RNNCell):
    def __init__(self, cell, output_size, activation=None, reuse=None):
        """Create a cell with output projection.
        Args:
            cell: an RNNCell, a projection to output_size is added to it.
            output_size: integer, the size of the output after projection.
            activation: (optional) an optional activation function.
            reuse: (optional) Python boolean describing whether to reuse variables
                in an existing scope.    If not `True`, and the existing scope already has
                the given variables, an error is raised.
        Raises:
            TypeError: if cell is not an RNNCell.
            ValueError: if output_size is not positive.
        """
        super(OutputProjectionWrapper, self).__init__(_reuse=reuse)
        if not _like_rnncell(cell):
            raise TypeError("The parameter cell is not RNNCell.")
        if output_size < 1:
            raise ValueError("Parameter output_size must be > 0: %d." % output_size)
        self._cell = cell
        self._output_size = output_size
        self._activation = activation
        self._linear = None

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        """Run the cell and output projection on inputs, starting from state."""
        output, res_state = self._cell(inputs, state)
        if self._linear is None:
            self._linear = _Linear(output, self._output_size, True)
        projected = self._linear(output)
        if self._activation:
            projected = self._activation(projected)
        return projected, res_state
