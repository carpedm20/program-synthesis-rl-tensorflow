import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import \
        Helper, TrainingHelper, BasicDecoder, BasicDecoderOutput, \
        BeamSearchDecoder, BeamSearchDecoderState
from tensorflow.contrib.rnn import \
        RNNCell, LSTMCell, MultiRNNCell, OutputProjectionWrapper
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import _beam_search_step

from .encoder import linear, int_shape

def decoder(num_examples, codes, code_lengths,
            encoder_out, dataset, config, mode):
    """ codes: [B, L]
        encoder_out: [BxN, 512]
    """
    dummy = {}
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
        dummy['embed'] = embed

        # [B, L, 256]
        code_embed = tf.nn.embedding_lookup(embed, codes)
        # [BxN, L, 256]
        tiled_code_embed = tf.tile(
                code_embed, [num_examples, 1, 1], name="tiled_code_embed")
        # [BxN]
        tiled_code_lengths = tf.tile(
                code_lengths, [num_examples], name="tiled_code_embed")
        # [BxN, 256]
        start_code_embed = tf.tile(
                [[0.0]], [batch_times_N, 256], name="start_token")

        # [BxN, L, 768]
        rnn_inputs = tf.concat(
                [tiled_encoder_out, tiled_code_embed], -1, name="rnn_inputs")
        # [BxN, 768]
        start_inputs = tf.expand_dims(tf.concat([
                encoder_out, start_code_embed], -1,
                name="start_inputs_concat"), 1, "start_inputs_concat")
        # [BxN, L+1, 768]
        rnn_inputs_shifted = tf.concat(
                [start_inputs, rnn_inputs], 1, name="rnn_inputs_shifted")

        decoder_cell = MultiRNNCell([
                LSTMCell(256),
                NaiveOutputProjectionWrapper(
                    MaxPoolWrapper(LSTMCell(256), num_examples),
                    num_token)], state_is_tuple=True)

        # [BxN, L, 256] -> [B, L, 256] #-> [BxN, L, V]
        decoder_logits, decoder_argamx, decoder_dummy = build_rnn(
                config, mode,
                cell=decoder_cell,
                rnn_train_inputs=rnn_inputs_shifted,
                start_code_embed=start_code_embed,
                batch_size=batch_times_N,
                target_lengths=tiled_code_lengths,
                embedding=embed,
                encoder_out=encoder_out,
                name="decoder_rnn")

        dummy.update(decoder_dummy)

        # [BxN, L, 256] -> [B, L, 256]
        decoder_logits = decoder_logits[:batch_size]
        decoder_argamx = decoder_argamx[:batch_size]

    if config.use_syntax:
        syntax_cell = MultiRNNCell([
                LSTMCell(256),
                OutputProjectionWrapper(
                    LSTMCell(256), num_examples)], state_is_tuple=True)

        # [B, L, 256]
        syntax_logits, syntax_argamx, syntax_dummy = build_rnn(
                config, mode,
                cell=syntax_cell,
                rnn_train_inputs=code_embed,
                start_code_embed=start_code_embed,
                batch_size=batch_size,
                target_lengths=code_lengths,
                name="syntax_rnn")

        dummy.update(syntax_dummy)

        exp_syntax_out = -tf.exp(syntax_out)
        decoder_logits = decoder_logits + syntax_logits

        dummy.update({
            'syntax_logits': syntax_logits,
            'syntax_argamx': syntax_argamx,
        })

    return decoder_logits, decoder_argamx, dummy

def build_rnn(config, mode, cell, rnn_train_inputs, start_code_embed,
              batch_size, target_lengths, embedding, encoder_out, name):

    dummy = {}
    if mode == 'train':
        helper = TrainingHelper(rnn_train_inputs, target_lengths)
    else:
        helper = GreedySamplingHelper(
                    batch_size, embedding, encoder_out, start_code_embed)

    initial_state = cell.zero_state(
            batch_size=batch_size, dtype=tf.float32)

    decoder = BasicDecoder(cell, helper, initial_state)

    (decoder_outputs, decoder_argmax), _, _ = \
            tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    maximum_iterations=config.max_code_length,
                    scope=name)

    if config.rl_mode > 0:
        helper = MultinomialSamplingHelper(
                batch_size, embedding, encoder_out, start_code_embed)

        if config.use_beam:
            decoder = CustomBeamSearchDecoder(
                    cell,
                    helper=helper,
                    embedding=embedding,
                    beam_width=config.beam_width,
                    encoder_out=encoder_out,
                    start_code_embed=start_code_embed,
                    initial_state=initial_state)
        else:
            decoder = BasicDecoder(
                    cell, helper, initial_state)

        (_, decoder_samples), _, _ = \
                tf.contrib.seq2seq.dynamic_decode(
                        decoder,
                        maximum_iterations=config.max_code_length,
                        scope=name + "_rl")

        if config.use_beam:
            # TODO: use top-1
            decoder_samples = decoder_samples.predicted_ids[:,:,0]

        dummy.update({ 'decoder_samples': decoder_samples })

    return decoder_outputs, decoder_argmax, dummy


class CustomBeamSearchDecoder(BeamSearchDecoder):
    def __init__(self,
                 cell,
                 helper,
                 embedding,
                 beam_width,
                 encoder_out,
                 start_code_embed,
                 initial_state,
                 end_token=0,
                 output_layer=None,
                 length_penalty_weight=0.0):

        self._cell = cell
        self._helper = helper
        self._output_layer = output_layer
        self._embedding_size = int_shape(embedding)[-1]

        self._initial_state = tuple(tf.nn.rnn_cell.LSTMStateTuple(
                tf.contrib.seq2seq.tile_batch(op[0], multiplier=beam_width),
                tf.contrib.seq2seq.tile_batch(op[1], multiplier=beam_width)) for op in initial_state)

        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                    lambda ids: tf.nn.embedding_lookup(embedding, ids))

        self._end_token = tf.convert_to_tensor(
                end_token, dtype=tf.int32, name="end_token")
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")

        # [BxN, 512]
        self._encoder_out = encoder_out
        # [BxN, 256]
        self._start_code_embed = start_code_embed

        # [BxN]
        self._batch_size = tf.shape(start_code_embed)[0]
        self._beam_width = beam_width
        self._length_penalty_weight = length_penalty_weight

        self._initial_cell_state = nest.map_structure(
                self._maybe_split_batch_beams,
                self._initial_state, self._cell.state_size)

        # [B, beam_width, E]
        _start_inputs = tf.concat([
                self._encoder_out, self._start_code_embed], -1,
                name="_start_inputs_concat")
        self._start_inputs = self.beam_tile(_start_inputs)
        self._finished = tf.zeros(
                [self._batch_size, self._beam_width], dtype=tf.bool)

    def beam_tile(self, tensor):
        return tf.tile(
                tf.expand_dims(tensor, 1), [1, self._beam_width, 1])

    def step(self, time, original_inputs, state, name=None):
        """Perform a decoding step.
        Args:
            time: scalar `int32` tensor.
            inputs: A (structure of) input tensors.
            state: A (structure of) state tensors and TensorArrays.
            name: Name scope for any created operations.
        Returns:
            `(outputs, next_state, next_inputs, finished)`.
        """
        batch_size = self._batch_size
        beam_width = self._beam_width
        end_token = self._end_token
        length_penalty_weight = self._length_penalty_weight

        batch_times_example, _, embed = int_shape(original_inputs)

        # inputs: [BxN (160), beam_width (64), 768]
        with tf.name_scope(name, "BeamSearchDecoderStep", (time, original_inputs, state)):
            cell_state = state.cell_state
            inputs = nest.map_structure(
                    lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), original_inputs)
            cell_state = nest.map_structure(
                    self._maybe_merge_batch_beams,
                    cell_state, self._cell.state_size)
            cell_outputs, next_cell_state = self._cell(inputs, cell_state)
            cell_outputs = nest.map_structure(
                    lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
            next_cell_state = nest.map_structure(
                    self._maybe_split_batch_beams,
                    next_cell_state, self._cell.state_size)

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            beam_search_output, beam_search_state = _beam_search_step(
                    time=time,
                    logits=cell_outputs,
                    next_cell_state=next_cell_state,
                    beam_state=state,
                    batch_size=batch_size,
                    beam_width=beam_width,
                    end_token=end_token,
                    length_penalty_weight=length_penalty_weight)

            finished = beam_search_state.finished
            sample_ids = beam_search_output.predicted_ids

            sampled_embed = tf.cond(
                    tf.reduce_all(finished),
                    lambda: self._start_inputs,
                    lambda: self._embedding_fn(sample_ids))

            sampled_embed.set_shape([None, beam_width, self._embedding_size])

            next_inputs = tf.concat(
                    [self.beam_tile(self._encoder_out), sampled_embed], -1)

        # inputs: [BxN (160), beam_width (64), 768]
        return (beam_search_output, beam_search_state, next_inputs, finished)


class GreedySamplingHelper(Helper):
    def __init__(self, batch_size, embedding, encoder_out, start_code_embed):
        # batch_times_N
        self._batch_size = batch_size
        self._encoder_out = encoder_out
        self._start_code_embed = start_code_embed

        self._embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))
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
        _start_inputs = tf.concat([
                self._encoder_out, self._start_code_embed], -1,
                name="_start_inputs_concat")
        return (finished, _start_inputs)

    def sample(self, time, outputs, state, name=None):
        del time, state
        sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        del time, outputs
        finished = tf.equal(sample_ids, self._end_token, name="finished")
        all_finished = tf.reduce_all(finished, name="all_finished")
        sampled_embed = tf.cond(
                all_finished,
                lambda: self._start_code_embed,
                lambda: self._embedding_fn(sample_ids), name="sampled_embed")
        next_inputs = tf.concat(
                [self._encoder_out, sampled_embed], -1,
                name="next_inputs_concat")
        return (finished, next_inputs, state)


class MultinomialSamplingHelper(GreedySamplingHelper):
    def __init__(self, batch_size, embedding, encoder_out, start_code_embed):
        super(MultinomialSamplingHelper, self).__init__(
                batch_size, embedding, encoder_out, start_code_embed)

    def sample(self, time, outputs, state, name=None):
        del time, state
        dist = tf.distributions.Categorical(logits=outputs)
        sample_ids = dist.sample()
        return sample_ids


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
