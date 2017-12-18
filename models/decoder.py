import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.seq2seq import TrainingHelper, InferenceHelper


class Decoder(object):
    def __init__(
            self, config,
            encoder_out, codes, dataset):

        embed = tf.get_variable(
                'embedding', [len(dataset.parser.tokens), 256], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5))

        import ipdb; ipdb.set_trace() 
        code_embed = tf.nn.embedding_lookup(embed, codes)

        rnn_cell = MultiRNNCell([
                ConcatOutputAndAttentionWrapper(encoder_out),
                LSTMCell(256),
                LSTMCell(256)], state_is_tuple=True, name="output_cell")

        decoder_rnn = build_rnn(input_cell, name="decoder_rnn")

        tf.map()

        if config.use_syntax:
            syntax_cell = MultiRNNCell([
                    LSTMCell(256),
                    LSTMCell(256)], state_is_tuple=True, name="syntax_cell")

            syntax_rnn = build_rnn(syntax_cell, name="syntax_rnn")

def build_rnn(cell, batch_size, name):
    decoder_init_state = cell.zero_state(
            batch_size=batch_size, dtype=tf.float32)

    if config.train:
        helper = TrainingHelper(inputs, sequence_length)
    else:
        helper = InferenceHelper()

    (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
            BasicDecoder(cell, helper, decoder_init_state), name=name)

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
