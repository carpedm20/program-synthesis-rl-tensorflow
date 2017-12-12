# Code based on https://github.com/alts/karel
import numpy as np
from collections import Counter

from hero import Hero


def draw2d(array):
    print("\n".join(["".join(["#" if val > 0 else "." for val in row]) for row in array]))

def border_mask(array, value):
    array[0,:], array[-1,:], array[:,0], array[:,-1] = value, value, value, value

class Karel(object):
    HERO_CHARS = '<^>v'
    MARKER_CHAR = 'o'
    WALL_CHAR = '#'
    EMPTY_CHAR = '.'

    def __init__(self, world_size=None, world_path=None, rng=None, max_marker_in_cell=1):
        if rng is None:
            self.rng = np.random.RandomState(123)
        else:
            self.rng = rng

        self.markers = []
        if world_path is not None:
            self.parse_world(world_path)
        elif world_size is not None:
            self.random_world(world_size, max_marker_in_cell)
        else:
            raise Exception(" [!] one of `world_size` and `world_path` should be passed")

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

    def random_world(self, world_size, max_marker_in_cell, wall_ratio=0.2, marker_ratio=0.3):
        height, width = world_size

        if height <= 2 or width <= 2:
            raise Exception(" [!] `height` and `width` should be larger than 2")

        # blank world
        self.world = np.chararray((height, width))
        self.world[:] = "."

        # wall
        wall_array = self.rng.rand(height, width)

        self.world[wall_array < wall_ratio] = "#"
        border_mask(self.world, "#")

        # hero
        x, y, direction = self.rng.randint(1, width-1), \
                self.rng.randint(1, height-1), self.rng.randint(4)
        self.hero = Hero((x, y), ((-1, 0), (1, 0), (0, -1), (0, 1))[direction])
        self.world[y, x] = '.'

        # markers
        marker_array = self.rng.rand(height, width)
        marker_array = (wall_array >= wall_ratio) & (marker_array < marker_ratio)
        border_mask(marker_array, False)

        self.markers = []
        for (y, x) in zip(*np.where(marker_array > 0)):
            self.markers.append((x, y))

        self.world = self.world.tolist()

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
                    if char in self.HERO_CHARS:
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

        for (x, y), count in Counter(self.markers).items():
            canvas[y][x] = str(count)

        canvas[self.hero.position[1]][self.hero.position[0]] = self.hero_char()

        print("\n".join("".join(row) for row in canvas))

    @property
    def state(self):
        """
            0: Hero facing North
            1: Hero facing South
            2: Hero facing West
            3: Hero facing East
            4: Wall
            5: 0 marker
            6: 1 marker
            7: 2 marker
            8: 3 marker
            9: 4 marker
            10: 5 marker
            11: 6 marker
            12: 7 marker
            13: 8 marker
            14: 9 marker
            15: 10 marker
        """
        state = self.zero_state.copy()
        state[:,:,5] = 1

        # 0 ~ 3: Hero facing North, South, West, East
        x, y = self.hero.position
        state[y, x, self.facing_idx] = 1

        # 4: wall or not
        for jdx, row in enumerate(self.world):
            for idx, char in enumerate(row):
                if char == self.WALL_CHAR:
                    state[jdx][idx][4] = 1
                elif char == self.WALL_CHAR or char in self.HERO_CHARS:
                    state[:,:,5] = 1

        # 5 ~ 15: marker counter
        for (x, y), count in Counter(self.markers).items():
            state[y][x][5] = 0
            state[y][x][5 + count] = 1

        # draw2d(state[:,:,5])
        return state

    def draw_exception(self, exception):
        pass

    def hero_char(self):
        # index will be in (-2, -1, 1, 2)
        index = self.hero.facing[0] + 2*self.hero.facing[1]
        return ' >v^<'[index]

    # Hero passthroughs
    def move(self):
        if not self.front_is_clear():
            #raise Exception('can\'t move. There is a wall in front of Hero')
            pass
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
            #raise Exception('can\'t pick marker from empty location')
            pass

    def put_marker(self):
        if not self.holding_markers():
            #raise Exception('can\'t put marker. Hero has none')
            pass
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

    @property
    def facing_idx(self):
        if self.facing_north:
            return 0
        elif self.facing_south:
            return 1
        elif self.facing_east:
            return 2
        elif self.facing_west:
            return 3
