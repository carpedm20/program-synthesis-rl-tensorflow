import os
import logging
import tensorflow as tf

from .decoder import decoder
from .encoder import encoder

logger = logging.getLogger("main")

def model(inputs, outputs, codes, code_lengths, dataset, config, train):
    num_examples = tf.shape(inputs)[1] # config.num_spec + config.num_heldout

    encoder_out = encoder(
            inputs, outputs, codes, dataset)
    decoder_out = decoder(
            num_examples, codes, code_lengths, encoder_out, dataset, config, train)
    return decoder_out

model_fn = tf.make_template('model', model)

class Model(object):
    def __init__(self, dataset, config, global_step, train):
        self.train = train
        self.config = config
        self.dataset = dataset
        self.global_step = global_step

        data = dataset.get_data('train' if train else 'test')

        self.inputs = data['inputs']
        self.outputs = data['outputs']
        self.codes = data['codes']
        self.code_lengths = data['code_lengths']
        self.iterator = data['iterator']

        self.inputs_without_heldout = self.inputs[:,:config.num_spec]
        self.outputs_without_heldout = self.outputs[:,:config.num_spec]

        if dataset.with_input_string:
            self.input_strings = data['input_strings']
            self.output_strings = data['output_strings']
        else:
            self.input_strings = None
            self.output_strings = None

        self.logits, self.argmax = model_fn(
                self.inputs_without_heldout,
                outputs=self.outputs_without_heldout,
                codes=self.codes,
                code_lengths=self.code_lengths,
                dataset=dataset,
                config=config,
                train=train)

        self.loss_mask = tf.sequence_mask(self.code_lengths, dtype=tf.float32)
        self.mle_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits,
                targets=self.codes,
                weights=self.loss_mask,
                name="MLE_loss")

        if config.use_rl:
            self.loss = None
        else:
            self.loss = self.mle_loss

        self.build_summary()

    def build_optim(self):
        self.optim = tf.train.AdamOptimizer(self.config.lr) \
                             .minimize(self.loss, self.global_step)

    def build_summary(self):
        max_summary = self.config.max_summary

        idx_to_code = lambda x: self.dataset.idx_to_text(x, markdown=True, beautify=False)
        idx_to_beatified_code= lambda x: self.dataset.idx_to_text(x, markdown=True, beautify=True)

        gt_codes = tf.py_func(
                idx_to_code, [self.codes[:max_summary]],
                tf.string, name="gt_codes")
        argmax_codes = tf.py_func(
                idx_to_code, [self.argmax[:max_summary]],
                tf.string, name="argmax_codes")

        gt_clean_codes = tf.py_func(
                idx_to_beatified_code, [self.codes[:max_summary]],
                tf.string, name="gt_clean_codes")
        argmax_clean_codes = tf.py_func(
                idx_to_beatified_code, [self.argmax[:max_summary]],
                tf.string, name="argmax_clean_codes")

        # Test spec + heldout examples 
        run = lambda x, y: self.dataset.run_and_test(x, y)
        outputs_of_argmax_codes = tf.py_func(
                run, [argmax_codes, self.inputs[:max_summary]],
                tf.string, name="outputs_of_argmax_codes")

        summaries = [
                tf.summary.scalar("loss/total", self.loss),
                tf.summary.scalar("loss/mle", self.mle_loss),
                tf.summary.text("clean_code_gt", gt_clean_codes),
                tf.summary.text("clean_code_argmax", argmax_clean_codes),
                tf.summary.text("code_gt", gt_codes),
                tf.summary.text("code_argmax", argmax_codes),
                tf.summary.text("outputs_pred_until_END", outputs_of_argmax_codes),
        ]
        if self.input_strings is not None:
            output_match = tf.cast(tf.equal(
                    outputs_of_argmax_codes,
                    self.output_strings[:max_summary]), tf.float32)

            def bool_to_str(x):
                array = x.astype(int).astype(str)
                array[array == '0'] = 'x'
                array[array == '1'] = 'o'
                return array

            output_match_strings = tf.py_func(bool_to_str, [output_match], tf.string)

            self.code_match = tf.cast(tf.equal(self.argmax, self.codes), tf.float32)
            self.code_match = tf.reduce_mean(
                    tf.minimum(self.code_match + (1 - self.loss_mask), 1))

            summaries.extend([
                    tf.summary.text("inputs_gt", self.input_strings[:max_summary]),
                    tf.summary.text("outputs_gt", self.output_strings[:max_summary]),
                    tf.summary.text("outputs_match", tf.cast(output_match_strings, tf.string)),

                    tf.summary.scalar("output_match/spec",
                                      tf.reduce_mean(output_match[:,:self.config.num_spec])),
                    tf.summary.scalar("output_match/heldout",
                                      tf.reduce_mean(output_match[:,:self.config.num_heldout])),
                    tf.summary.scalar("output_match/total",
                                      tf.reduce_mean(output_match)),

                    tf.summary.scalar("code/match", self.code_match),
                    #tf.summary.scalar("metric/generalization", tf.reduce_mean(output_match)),
            ])

        self.summary_op = tf.summary.merge(summaries)

        summary_path = os.path.join(
                self.config.model_path, 'train' if self.train else 'test')
        self.writer = tf.summary.FileWriter(summary_path)

    def run(self, sess, with_update=True, with_summary=False):
        fetches = { 'step': self.global_step }

        if with_update:
            fetches['optim'] = self.optim
        if with_summary:
            fetches['summary'] = self.summary_op
            fetches['loss'] = self.loss

        out = sess.run(fetches)
        step = out['step']

        if with_summary:
            self.writer.add_summary(out['summary'], step)
            logger.info("[INFO] loss: {:.4f}".format(out['loss']))

        return step
