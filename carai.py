
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
)

from config import SCALE_FACTOR_X, SCALE_FACTOR_Y, ROAD_WIDTH, ROAD_HEIGHT
from math import floor
import sys

def get_idx(y, x):
    return floor(y / ROAD_WIDTH), floor(x / ROAD_HEIGHT)

class CarAI:

    def __init__(self, car, road_collection, roadmap, dest):
        self.car = car
        self.visited_cells = set()
        self.pressed_keys = {
            K_UP: False,
            K_DOWN: False,
            K_LEFT: False,
            K_RIGHT: False,
        }
        self.counter = 0
        self.road_collection = road_collection
        self.entering_from = { # maps present direction with possible entry
            K_UP: 4,
            K_DOWN: 3,
            K_LEFT: 2,
            K_RIGHT: 1
        }
        self.roadmap = {}
        for a, b in roadmap:
            self.roadmap[a] = b
        self.dest = dest
        self.keymap = {
            K_UP: "up",
            K_DOWN: "down",
            K_LEFT: "left",
            K_RIGHT: "right"
        }
        self.present_cell = None
        self.current_direction = K_UP

    def calculate_max_distance(self, direction, default_max, start, stop, step):
        direction = self.entering_from[direction]
        for end in range(start, stop, step):
            road = self.road_collection[end]
            if road == None or (direction not in self.roadmap[road.roadid]):
                return abs((start - end) // step)
        return default_max

    def calculate_next_move(self, debug=True):
        self.pressed_keys[self.current_direction] = False
        base_i, base_j = get_idx(*self.car.rect.center)
        if self.present_cell == self.dest:
            return self.pressed_keys
        if self.present_cell == (base_i, base_j):
            self.pressed_keys[self.current_direction] = True
            return self.pressed_keys
        self.visited_cells.add((base_i, base_j))
        self.present_cell = (base_i, base_j)

        max_right = self.calculate_max_distance(K_RIGHT, SCALE_FACTOR_Y - base_i - 1,
            (base_i + 1) * SCALE_FACTOR_X + base_j, (SCALE_FACTOR_Y - 1) * SCALE_FACTOR_X + base_j + 1,
                SCALE_FACTOR_X)
        max_left = self.calculate_max_distance(K_LEFT, base_i,
                (base_i - 1) * SCALE_FACTOR_X + base_j, -1,
                -SCALE_FACTOR_X)
        max_up = self.calculate_max_distance(K_UP, base_j,
                base_i * SCALE_FACTOR_X + base_j - 1, base_i * SCALE_FACTOR_X - 1, -1)
        max_down = self.calculate_max_distance(K_DOWN, SCALE_FACTOR_X - base_j - 1,
                base_i * SCALE_FACTOR_X + base_j + 1, (base_i + 1) * SCALE_FACTOR_X, 1)
        if debug:
            print('\b' * 85, self.counter,  "left:", max_left, "right:", max_right,
                  "up:", max_up, "down:", max_down, end='')

        can_move = [(K_RIGHT, max_right, max_right, 0, self.keymap[K_RIGHT]),
                    (K_LEFT, max_left, -max_left, 0, self.keymap[K_LEFT]),
                    (K_UP, max_up, 0, -max_up, self.keymap[K_UP]),
                    (K_DOWN, max_down, 0, max_down, self.keymap[K_DOWN])]
        #print("\ncan_move:", can_move)
        can_move = sorted(can_move, key=lambda x: x[1])
        #print("can_move(sorted):", can_move)
        can_move = list(filter(lambda x: (base_i + x[2], base_j + x[3]) not in self.visited_cells,
                               can_move))
        #print("can_move(filtered):", can_move)
        if len(can_move) > 0:
            key = can_move[-1][0]
            if debug:
                print(" going", "%-5s" % self.keymap[key], end='')
            self.pressed_keys[key] = True
            self.current_direction = key
            self.counter += 1
        else:
            if debug:
                print(" no possible move found")
            self.present_cell = self.dest
        if debug:
            sys.stdout.flush()
        return self.pressed_keys
