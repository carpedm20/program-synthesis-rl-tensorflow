from encoder import encoder
from decoder import decoder

class Model(object):
    def __init__(self, config, inputs):
        self.config = config

        self.enc = encoder(inputs)
        self.dec = encoder(inputs)
