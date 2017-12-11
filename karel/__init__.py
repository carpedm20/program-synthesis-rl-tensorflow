# Code based on https://github.com/alts/karel
import numpy as np
from hero import Hero

class LogicException(Exception):
    pass

class Karel(object):
    KAREL_CHARS = '<^>v'
    MARKER_CHAR = 'o'

    def __init__(self, map_path):
        self.map = None
        self.hero = None
        self.screen = None
        self.markers = []
        self.construct_map(map_path)

    def __enter__(self):
        self.start_screen()
        return self

    def __exit__(self, *args):
        self.end_screen()

    def start_screen(self):
        pass

    def end_screen(self):
        pass

    def construct_map(self, map_path):
        directions = {
            '>': (1, 0),
            '^': (0, 1),
            '<': (-1, 0),
            'v': (0, -1),
        }

        map = [[]]
        with open(map_path, 'rb') as f:
            for y, line in enumerate(f):
                row = []
                for x, char in enumerate(line.strip()):
                    if char in self.KAREL_CHARS:
                        self.hero = Hero((x + 1, y + 1), directions[char])
                        char = '.'
                    elif char == self.MARKER_CHAR:
                        self.markers.append((x + 1, y + 1))
                        char = '.'
                    row.append(char)
                map.append(['#'] + row + ['#'])

        map.append([])
        for _ in xrange(len(map[1])):
            map[0].append('#')
            map[-1].append('#')
        self.map = map

    def draw(self):
        canvas = np.array(self.map)

        for bx, by in self.markers:
            canvas[by][bx] = 'o'

        canvas[self.hero.position[1]][self.hero.position[0]] = self.hero_char()

        print("\n".join("".join(row) for row in canvas))

    def get_state(self):
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
        states = [
                self.facing_north, self.facing_south,
                self.facing_west, self.facing_east,
        ]
        state = self.zeros_like(self.map)

        for jdx, row in enumerate(self.map):
            for idx, point in enumerate(row):
                self.map[jdx][idx]

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
        return self.map[next_y][next_x] == '.'

    def left_is_clear(self):
        next_x = self.hero.position[0] + self.hero.facing[1]
        next_y = self.hero.position[1] - self.hero.facing[0]
        return self.map[next_y][next_x] == '.'

    def right_is_clear(self):
        next_x = self.hero.position[0] - self.hero.facing[1]
        next_y = self.hero.position[1] + self.hero.facing[0]
        return self.map[next_y][next_x] == '.'

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

