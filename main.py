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

    trainer = Trainer(config, rng)

    if not config.test:
        trainer.train()
    else:
        if not config.map:
            raise Exception("[!] You should specify `map` to synthesize a program")
        trainer.synthesize(config.map)

if __name__ == "__main__":
    config, unparsed = get_config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
