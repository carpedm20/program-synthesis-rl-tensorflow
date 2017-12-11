#!/usr/bin/env python
import sys
from . import Karel

if __name__ == '__main__':
    if len(sys.argv) > 2:
        code_path = sys.argv[1]
        map_path = sys.argv[2]
    else:
        print(" [*] Usage: ./interpreter.py KAREL MAP")
        sys.exit()

    karel = Karel(map_path)

    def exec_and_draw(fn):
        def inner():
            res = fn()
            karel.draw()
            return res
        return inner

    move = exec_and_draw(karel.move)
    turn_left = exec_and_draw(karel.turn_left)
    pick_marker = exec_and_draw(karel.pick_marker)
    put_marker = exec_and_draw(karel.put_marker)

    front_is_clear = karel.front_is_clear
    right_is_clear = karel.right_is_clear
    left_is_clear = karel.left_is_clear
    facing_north = karel.facing_north
    facing_south = karel.facing_south
    facing_east = karel.facing_east
    facing_west = karel.facing_west
    marker_is_present = karel.marker_is_present

    with open(code_path) as f:
        exec(f.read())
