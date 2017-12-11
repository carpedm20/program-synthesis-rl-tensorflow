import sys
import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config

config = None

def main(_):
    prepare_dirs(config)

    rng = np.random.RandomState(config.seed)
    tf.set_random_seed(config.seed)

    trainer = Trainer(config, rng)
    save_config(config.model_dir, config)

    if not config.test:
        trainer.train()
    else:
        if not config.map:
            raise Exception("[!] You should specify `map` to synthesize a program")
        trainer.synthesize(config.map)

if __name__ == "__main__":
    config, unparsed = get_config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
