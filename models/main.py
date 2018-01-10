import os
import logging
from glob import glob
import tensorflow as tf

from .decoder import decoder
from .encoder import encoder, int_shape

logger = logging.getLogger("main")


def model(inputs, outputs, codes, code_lengths,
          dataset, config, mode):
    num_examples = tf.shape(inputs)[1] # config.num_spec + config.num_heldout

    encoder_out = encoder(
            inputs, outputs, codes, dataset)
    decoder_out = decoder(
            num_examples, codes, code_lengths, encoder_out, dataset, config, mode)
    return decoder_out

model_fn = tf.make_template('model', model)


class Model(object):
    def __init__(self, dataset, config, global_step, mode):
        self.mode = mode
        self.config = config
        self.dataset = dataset
        self.global_step = global_step

        data = dataset.get_data()

        self.inputs_with_heldout = data['inputs']
        self.outputs_with_heldout = data['outputs']
        self.codes = data['codes']
        self.code_lengths = data['code_lengths']

        self.inputs_without_heldout = self.inputs_with_heldout[:,:config.num_spec]
        self.outputs_without_heldout = self.outputs_with_heldout[:,:config.num_spec]

        self.idx_to_code = lambda x: \
                self.dataset.idx_to_text(x, markdown=True, beautify=False)
        self.idx_to_beatified_code = lambda x: \
                self.dataset.idx_to_text(x, markdown=True, beautify=True)

        if dataset.with_input_string:
            self.input_strings = data['input_strings']
            self.output_strings = data['output_strings']
        else:
            self.input_strings = None
            self.output_strings = None

        self.summaries = []
        self._build_model()

    def _build_model(self):
        # [32, 13, 52], [32, 13]
        self.logits, self.argmax, self.dummy = model_fn(
                self.inputs_without_heldout, # [B, E, 8, 8, 16]
                outputs=self.outputs_without_heldout,
                codes=self.codes, # [B, L]
                code_lengths=self.code_lengths, # [B]
                dataset=self.dataset,
                config=self.config,
                mode=self.mode)

        # [B, L]
        self.loss_mask = tf.sequence_mask(
                self.code_lengths, dtype=tf.float32)

        # if rl is used, every batch is needed to be calculated
        if self.config.rl_mode:
            max_summary = None
        else:
            max_summary = self.config.max_summary

        self.argmax_codes = tf.py_func(
                self.idx_to_code, [self.argmax[:max_summary]],
                tf.string, name="argmax_codes")

        # Test spec + heldout examples 
        run = lambda x, y: self.dataset.run_with_example(x, y, with_test=True)
        self.outputs_of_argmax_codes, self.timeout_count, \
                self.syntax_error_count, self.be_able_to_run_count, \
                self.invalid_move_count = tf.py_func(
                    run, [
                        self.argmax_codes[:max_summary],
                        self.inputs_with_heldout[:max_summary]
                    ],
                    [tf.string, tf.float32, tf.float32, tf.float32, tf.float32],
                    name="outputs_of_argmax_codes")

        self.argmax_output_match = tf.equal(
                self.outputs_of_argmax_codes,
                self.output_strings[:max_summary])

        self.loss = None
        if self.mode == 'train':
            # [384 = 32 * 13,52] vs [1344] = [32 * 42]
            self.mle_loss = tf.contrib.seq2seq.sequence_loss(
                    logits=self.logits, # [32(B), 13, 52(V)] <-> [32, 42, 52]
                    targets=self.codes, # [32(B), 42(L)]
                    weights=self.loss_mask, # [32(B), 42(L)]
                    name="MLE_loss")

            if self.config.rl_mode > 0:
                self._build_rl()

                self.loss = self.pi_loss
            else:
                self.loss = self.mle_loss

        self._build_summary()

    def _build_rl(self):
        max_summary = self.config.max_summary
        self.samples = self.dummy['decoder_samples']

        self.sampled_codes = tf.py_func(
                self.idx_to_code, [self.samples],
                tf.string, name="sampled_codes")

        self.clean_sampled_codes = tf.py_func(
                self.idx_to_beatified_code, [self.samples[:max_summary]],
                tf.string, name="clean_sampled_codes")

        self.summaries.append(
                tf.summary.text("clean_code_sampled", self.clean_sampled_codes))

        # Test spec + heldout examples 
        run = lambda x, y: self.dataset.run_with_example(x, y, with_test=False)

        self.sampled_codes_outputs = tf.py_func(
                    run, [
                        self.sampled_codes,
                        self.inputs_with_heldout,
                    ],
                    tf.string, name="sampled_codes_outputs")

        self.sampled_output_match = tf.equal(
                self.sampled_codes_outputs,
                self.output_strings,
                name='sampled_output_match')

        # +1 if the program matches all samples, including the held out one
        # and 0 otherwise.
        # [B]
        self.sampled_reward = tf.cast(tf.reduce_all(
                self.sampled_output_match, -1), tf.float32)

        self.log_prob = tf.nn.log_softmax(self.logits)
        self.prob = tf.nn.softmax(self.logits)
        self.entropy = -tf.reduce_sum(self.prob * self.log_prob)

        if self.config.rl_mode == 2:
            self.argmax_reward = tf.cast(tf.reduce_all(
                    self.argmax_output_match, -1), tf.float32)
            reward = self.normalized_reward = self.argmax_reward - self.sampled_reward

            self.summaries.extend([
                    tf.summary.scalar("rl/argmax_reward", 
                                    tf.reduce_mean(self.argmax_reward)),
                    tf.summary.scalar("rl/normalized_reward", 
                                    tf.reduce_mean(self.normalized_reward)),
            ])
        else:
            reward = self.sampled_reward

        self.argmax_one_hot = tf.one_hot(
                self.argmax, self.dataset.num_token)

        # [B, L, V] -> [B, L]
        self.log_action_prob = tf.reduce_sum(
                self.log_prob * self.argmax_one_hot, -1)

        self.pi_loss = -tf.reduce_sum(
                self.log_action_prob * tf.expand_dims(reward, -1))

        self.pi_loss -= self.entropy * self.config.entropy_penalty

        self.summaries.extend([
                tf.summary.scalar("rl/sampled_reward", 
                                  tf.reduce_mean(self.sampled_reward)),
                tf.summary.scalar("rl/pi_loss", 
                                  tf.reduce_mean(self.pi_loss)),
                tf.summary.scalar("rl/entropy", self.entropy),
        ])


    def _build_summary(self):
        max_summary = self.config.max_summary

        gt_codes = tf.py_func(
                self.idx_to_code, [self.codes[:max_summary]],
                tf.string, name="gt_codes")

        gt_clean_codes = tf.py_func(
                self.idx_to_beatified_code, [self.codes[:max_summary]],
                tf.string, name="gt_clean_codes")
        argmax_clean_codes = tf.py_func(
                self.idx_to_beatified_code, [self.argmax[:max_summary]],
                tf.string, name="argmax_clean_codes")

        self.summaries.extend([
                tf.summary.scalar("code/timeout_ratio",
                                  tf.reduce_mean(self.timeout_count)),
                tf.summary.scalar("code/syntax_error_ratio",
                                  tf.reduce_mean(self.syntax_error_count)),
                tf.summary.scalar("code/invalid_move_ratio",
                                  tf.reduce_mean(self.invalid_move_count)),
                tf.summary.scalar("code/be_able_to_run_ratio",
                                  tf.reduce_mean(self.be_able_to_run_count)),

                tf.summary.text("clean_code_gt", gt_clean_codes),
                tf.summary.text("clean_code_argmax", argmax_clean_codes),
                #tf.summary.text("code_gt", gt_codes),
                #tf.summary.text("code_argmax", self.argmax_codes),
                tf.summary.text("outputs_pred_until_END", self.outputs_of_argmax_codes),
        ])

        if self.mode == 'train':
            self.summaries.extend([
                    tf.summary.scalar("loss/total", self.loss),
                    tf.summary.scalar("loss/mle", self.mle_loss),
            ])

        if self.input_strings is not None:
            def bool_to_str(x):
                array = x.astype(int).astype(str)
                array[array == '0'] = 'x'
                array[array == '1'] = 'o'
                return array

            argmax_output_match_float = tf.cast(
                    self.argmax_output_match[:max_summary], tf.float32)

            argmax_output_match_strings = tf.py_func(
                    bool_to_str, [argmax_output_match_float], tf.string)

            if self.mode != 'train':
                self.argmax_length = tf.shape(self.argmax)[1]
                self.code_length = tf.shape(self.codes)[1]

                self.argmax_m_code = self.argmax_length - self.code_length
                self.code_m_argmax = self.code_length - self.argmax_length

                self.argmax_pad = [[0, 0], [0, tf.maximum(0, self.code_m_argmax)]]
                self.code_pad = [[0, 0], [0, tf.maximum(0, self.argmax_m_code)]]

                self.argmax = tf.pad(self.argmax, self.argmax_pad, "CONSTANT")
                self.codes = tf.pad(self.codes, self.code_pad, "CONSTANT")

                neg_loss_mask = 1 - tf.pad(self.loss_mask, self.code_pad, "CONSTANT")
            else:
                neg_loss_mask = 1 - self.loss_mask

            self.code_match = tf.cast(tf.equal(self.argmax, self.codes), tf.float32)
            masked_code_match = tf.minimum(self.code_match + neg_loss_mask, 1)

            code_match_ratio = tf.reduce_mean(masked_code_match)
            code_exact_match = tf.equal(tf.reduce_mean(masked_code_match, [-1]), 1)
            code_exact_match_ratio = tf.reduce_mean(tf.cast(code_exact_match, tf.float32))

            self.summaries.extend([
                    tf.summary.text(
                            "inputs_gt",
                            self.input_strings[:max_summary]),
                    tf.summary.text(
                            "outputs_gt",
                            self.output_strings[:max_summary]),
                    tf.summary.text(
                            "outputs_match",
                            tf.cast(argmax_output_match_strings, tf.string)),

                    tf.summary.scalar(
                            "output_match/spec",
                            tf.reduce_mean(
                                argmax_output_match_float[:,:self.config.num_spec])),
                    tf.summary.scalar(
                            "output_match/heldout",
                            tf.reduce_mean(
                                argmax_output_match_float[:,:self.config.num_heldout])),
                    tf.summary.scalar(
                            "output_match/total",
                            tf.reduce_mean(argmax_output_match_float)),

                    tf.summary.scalar(
                            "code/match_ratio", code_match_ratio),
                    tf.summary.scalar(
                            "code/exact_match_ratio", code_exact_match_ratio),
                    #tf.summary.scalar("metric/generalization", tf.reduce_mean(argmax_output_match)),
            ])

        self.summary_op = tf.summary.merge(self.summaries)

        summary_path = os.path.join(
                self.config.model_path, self.mode)
        self.writer = tf.summary.FileWriter(summary_path)

    def run(self, sess, with_update=True, with_summary=False):
        fetches = { 'step': self.global_step }

        if with_update:
            fetches['optim'] = self.optim
        if with_summary:
            fetches['summary'] = self.summary_op
            if self.loss is not None:
                fetches['loss'] = self.loss

        out = sess.run(fetches)
        step = out['step']

        if with_summary:
            self.writer.add_summary(out['summary'], step)
            if self.loss is not None:
                logger.info("[INFO] loss: {:.4f}".format(out['loss']))

        return step

    def build_optim(self):
        self.optim = tf.train.AdamOptimizer(self.config.lr) \
                             .minimize(self.loss, self.global_step)
        self.saver = tf.train.Saver(
                max_to_keep=5, keep_checkpoint_every_n_hours=4)

    def restore(self, sess):
        checkpoint_path = \
                get_most_recent_checkpoint(self.config.pretrain_path)
        logger.info('Load checkpoint from: {}'.format(checkpoint_path))
        self.saver.restore(sess, checkpoint_path)

    def save(self, sess, step, path=None):
        if path is None:
            path = "{}/{}".format(self.config.model_path, self.mode)
        logger.info('Saving checkpoint to: {}-{}'.format(path, step))
        self.saver.save(sess, path, global_step=step)


def get_most_recent_checkpoint(checkpoint_dir_or_path):
    if os.path.isdir(checkpoint_dir_or_path):
        lastest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir_or_path)
    else:
        lastest_checkpoint = checkpoint_dir_or_path
    return lastest_checkpoint
