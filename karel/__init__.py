# Code based on https://github.com/alts/karel
import numpy as np
from hero import Hero

class LogicException(Exception):
    pass

class Karel(object):
    KAREL_CHARS = '<^>v'
    MARKER_CHAR = 'o'
    WALL_CHAR = '#'
    EMPTY_CHAR = '.'

    def __init__(self, word=None, world_path=None):
        self.world = None
        self.hero = None
        self.screen = None
        self.markers = []

        if world_path is not None:
            self.parse_world(world_path)
        elif word is not None:
            self.word = word
        else:
            raise Exception(" [!] one of `word` and `world_path` should be passed")

        state = np.zeros_like(self.world, dtype=int)
        self.zero_state = np.tile(np.expand_dims(state, -1), [1, 1, 16])

    def __enter__(self):
        self.start_screen()
        return self

    def __exit__(self, *args):
        self.end_screen()

    def start_screen(self):
        pass

    def end_screen(self):
        pass

    def parse_world(self, world_path):
        directions = {
            '>': (1, 0),
            '^': (0, 1),
            '<': (-1, 0),
            'v': (0, -1),
        }

        world = [[]]
        with open(world_path, 'rb') as f:
            for y, line in enumerate(f):
                row = []
                for x, char in enumerate(line.strip()):
                    if char in self.KAREL_CHARS:
                        self.hero = Hero((x + 1, y + 1), directions[char])
                        char = '.'
                    elif char == self.MARKER_CHAR:
                        self.markers.append((x + 1, y + 1))
                        char = '.'
                    elif char.isdigit():
                        for _ in range(int(char)):
                            self.markers.append((x + 1, y + 1))
                        char = '.'
                    elif char in [self.WALL_CHAR, self.EMPTY_CHAR]:
                        pass
                    else:
                        raise Exception(" [!] `{}` is not a valid character".format(char))
                    row.append(char)
                world.append(['#'] + row + ['#'])

        world.append([])
        for _ in xrange(len(world[1])):
            world[0].append('#')
            world[-1].append('#')

        self.world = world

    def draw(self):
        canvas = np.array(self.world)

        for bx, by in self.markers:
            canvas[by][bx] = 'o'

        canvas[self.hero.position[1]][self.hero.position[0]] = self.hero_char()

        print("\n".join("".join(row) for row in canvas))

    @property
    def state(self):
        """
            Hero facing North
            Hero facing South
            Hero facing West
            Hero facing East
            Obstacle
            0 marker
            1 marker
            2 marker
            3 marker
            4 marker
            5 marker
            6 marker
            7 marker
            8 marker
            9 marker
            10 marker
        """
        state = self.zero_state.copy()

        # Hero facing North, South, West, East
        facing_idx = self.hero.facing[0] * 2 + self.hero.facing[1]
        state[self.hero.position][facing_idx] = 1

        states = [
                self.facing_north, self.facing_south,
                self.facing_west, self.facing_east,
        ]

        for jdx, row in enumerate(self.world):
            for idx, point in enumerate(row):
                self.world[jdx][idx]

    def draw_exception(self, exception):
        pass

    def hero_char(self):
        # index will be in (-2, -1, 1, 2)
        index = self.hero.facing[0] + 2*self.hero.facing[1]
        return ' >v^<'[index]

    # Hero passthroughs
    def move(self):
        if not self.front_is_clear():
            raise LogicException('can\'t move. There is a wall in front of Hero')
        self.hero.move()

    def turn_left(self):
        self.hero.turn_left()

    def pick_marker(self):
        position = self.hero.position
        for i, coord in enumerate(self.markers):
            if coord == self.hero.position:
                del self.markers[i]
                self.hero.pick_marker()
                break
        else:
            raise LogicException('can\'t pick marker from empty location')

    def put_marker(self):
        if not self.holding_markers():
            raise LogicException('can\'t put marker. Hero has none')
        self.markers.append(self.hero.position)
        self.hero.put_marker()

    # world conditions
    def front_is_clear(self):
        next_x = self.hero.position[0] + self.hero.facing[0]
        next_y = self.hero.position[1] + self.hero.facing[1]
        return self.world[next_y][next_x] == '.'

    def left_is_clear(self):
        next_x = self.hero.position[0] + self.hero.facing[1]
        next_y = self.hero.position[1] - self.hero.facing[0]
        return self.world[next_y][next_x] == '.'

    def right_is_clear(self):
        next_x = self.hero.position[0] - self.hero.facing[1]
        next_y = self.hero.position[1] + self.hero.facing[0]
        return self.world[next_y][next_x] == '.'

    def marker_is_present(self):
        return self.hero.position in self.markers

    def holding_markers(self):
        return self.hero.holding_markers()

    @property
    def facing_north(self):
        return self.hero.facing[1] == -1

    @property
    def facing_south(self):
        return self.hero.facing[1] == 1

    @property
    def facing_east(self):
        return self.hero.facing[0] == 1

    @property
    def facing_west(self):
        return self.hero.facing[0] == -1
