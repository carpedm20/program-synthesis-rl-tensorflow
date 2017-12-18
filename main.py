import sys
import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from utils import prepare, set_random_seed

config = None

def main(_):
    prepare(config)
    rng = set_random_seed(config.seed)

    sess_config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True)
    sess_config.gpu_options.allow_growth=True

    trainer = Trainer(config, rng)

    with tf.train.MonitoredTrainingSession(
            is_chief=True,
            checkpoint_dir=config.model_path,
            #hooks=[ tf.train.SummarySaverHook() ],
            config=sess_config) as sess:
        while not sess.should_stop():
            if config.train:
                trainer.train(sess)
            else:
                if not config.map:
                    raise Exception("[!] You should specify `map` to synthesize a program")
                trainer.synthesize(sess, config.map)

if __name__ == "__main__":
    config, unparsed = get_config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
