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
        token_num = dataset.token_num

        with tf.variable_scope("decoder"):
            # [BxN, L, 512]
            tiled_cnn_out = tf.tile(
                    tf.expand_dims(encoder_out, 1),
                    [1, tf.shape(codes)[1], 1],
                    name="tiled_cnn_out")

            embed = tf.get_variable(
                    'embedding', [len(dataset.parser.tokens), 256], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.5))

            # [B, L, 256]
            code_embed = tf.nn.embedding_lookup(embed, codes)
            # [BxN, L, 256]
            tiled_code_embed = tf.tile(code_embed, [2, 1, 1], name="tiled_code_embed")

            # [BxN, L, 768]
            rnn_input = tf.concat([code_embed, tiled_cnn_out], -1)

            shape = tf.shape(rnn_input)
            batch_times_N, L, input_dim = shape[0], shape[1], shape[2]

            decoder_cell = MultiRNNCell([
                    LSTMCell(256),
                    LSTMCell(256)], state_is_tuple=True)

            # [BxN, L, 256]
            tiled_decoder_out = build_rnn(
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

            decoder_logits = linear(max_pool, token_num, "out")

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

            syntax_logits = linear(max_pool, token_num, "out")

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


class ConcatOutputAndAttentionWrapper(RNNCell):
    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)
