import os
import json
import errno
import signal
import logging
import numpy as np
import tensorflow as tf
from functools import wraps
from datetime import datetime
from pyparsing import nestedExpr

logger = logging.getLogger("main")


def get_time():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def str2bool(v):
    return v.lower() in ('true', '1')

class Tcolors:
    CYAN = '\033[1;30m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class TimeoutError(Exception):
    pass

def timeout_fn(timeout=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout) #used timer instead of alarm
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    return decorator

def beautify_fn(inputs, indent=1, tabspace=2):
    lines, queue = [], []
    space = tabspace * " "

    for item in inputs:
        if item == ";":
            lines.append(" ".join(queue))
            queue = []
        elif type(item) == str:
            queue.append(item)
        else:
            lines.append(" ".join(queue + ["{"]))
            queue = []

            inner_lines = beautify_fn(item, indent=indent+1, tabspace=tabspace)
            lines.extend([space + line for line in inner_lines[:-1]])
            lines.append(inner_lines[-1])

    if len(queue) > 0:
        lines.append(" ".join(queue))

    return lines + ["}"]

def pprint(code, *args, **kwargs):
    print(beautify(code, *args, **kwargs))

def beautify(code, tabspace=2):
    array = nestedExpr('{','}').parseString("{"+code+"}").asList()
    lines = beautify_fn(array[0])
    return "\n".join(lines[:-1]).replace(' ( ', '(').replace(' )', ')')

def makedirs(path):
    if not os.path.exists(path):
        logger.info("[MAKE] directory: {}".format(path))
        os.makedirs(path)

def get_rng(rng, seed=123):
    if rng is None:
        rng = np.random.RandomState(seed)
    return rng

def load_config(config, skip_list=[]):
    config_keys = vars(config).keys()
    config_path = os.path.join(config.load_path, PARAMS_NAME)

    with open(path) as fp:
        new_config = json.load(fp)

    for key, value in new_config.items():
        if key in skip_list:
            continue

        original_value = getattr(config, key)
        if original_value != value:
            logger.info("[UPDATE] config `{}`: {} -> {}".format(key, getattr(config, key), value))
            setattr(config, key, value)

def save_config(config, config_filename='config.json'):
    config_path = os.path.join(config.model_path, config_filename)
    with open(config_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)
    logger.info('[SAVE] config: {}'.format(config_path))

def prepare(config):
    formatter = logging.Formatter(
            "%(levelname)s:%(asctime)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(tf.logging.INFO)

    if config.model_path is None:
        config.model_name = "{}_{}".format(config.tag, get_time())
        config.model_path = os.path.join(config.base_dir, config.model_name)

        makedirs(config.model_path)
        save_config(config)
    else:
        if not config.model_path.startswith(config.base_dir):
            new_path = os.path.join(config.base_dir, new_path)
            setattr(config, 'model_path', new_path)
        logger.info("[SET] model_path: {}".format(config.model_path))

def set_random_seed(seed):
    tf.set_random_seed(seed)
    rng = np.random.RandomState(seed)
    return rng
