# Code based on https://github.com/alts/karel

class Hero(object):
    def __init__(self, position, facing, beeper_bag=None):
        self.position = position
        self.facing = facing
        self.beeper_bag = None

    def move(self):
        self.position = (
            self.position[0] + self.facing[0],
            self.position[1] + self.facing[1]
        )

    def turn_left(self):
        self.facing = (
            self.facing[1],
            -self.facing[0]
        )

    def holding_beepers(self):
        return (self.beeper_bag is None) or self.beeper_bag > 0

    def pick_beeper(self):
        if self.beeper_bag is not None:
            self.beeper_bag += 1

    def put_beeper(self):
        if self.beeper_bag is not None:
            self.beeper_bag -= 1
