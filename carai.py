
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
)

from config import SCALE_FACTOR_X, SCALE_FACTOR_Y, ROAD_WIDTH, ROAD_HEIGHT
from math import floor
from car import Car
import sys

def get_idx(y, x):
    return floor(y / ROAD_WIDTH), floor(x / ROAD_HEIGHT)

class CarAI:

    def __init__(self, road_collection, roadmap, dest):
        self.car = Car()
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


class CarAI_NN:

    def __init__(self, road_collection, roadmap, dest, network):
        self.car = Car()
        self.visited_cells = set()
        self.pressed_keys = {
            K_UP: False,
            K_DOWN: False,
            K_LEFT: False,
            K_RIGHT: False,
        }
        self.counter = 1
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
        self.network = network
        self.score = None
        self.prev_can_moves = None
        self.prev_choice = None
        self.alive_timer = 1

    def calculate_max_distance(self, direction, default_max, start, stop, step):
        direction = self.entering_from[direction]
        for end in range(start, stop, step):
            road = self.road_collection[end]
            if road == None or (direction not in self.roadmap[road.roadid]):
                return abs((start - end) // step)
        return default_max

    def calculate_score(self):
        # score is calculated based on the sum of the absolute
        # differences of the expected and received output
        expected_output = [0] * 4
        output = [x[1] for x in self.prev_can_moves]
        i = max(range(len(output)), key=output.__getitem__)
        expected_output[i] = 1
        score = 0
        for a, b in zip(expected_output, self.prev_choice):
            score += abs(a - b)
        self.score = ((1/score) * 0.5) + (self.alive_timer * 0.5)

    def __hash__(self):
        return hash(self.score)

    def calculate_next_move(self, time_passed, debug=False):
        self.pressed_keys[self.current_direction] = False
        if self.car.killed:
            if self.score == None:
                self.calculate_score()
            return self.pressed_keys
        base_i, base_j = get_idx(*self.car.rect.center)
        if self.present_cell == self.dest:
            return self.pressed_keys
        self.alive_timer += time_passed
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

        can_move = [[K_LEFT, max_left, -max_left, 0, self.keymap[K_LEFT]],
                    [K_RIGHT, max_right, max_right, 0, self.keymap[K_RIGHT]],
                    [K_UP, max_up, 0, -max_up, self.keymap[K_UP]],
                    [K_DOWN, max_down, 0, max_down, self.keymap[K_DOWN]]]
        # we don't sort anymore, we leave this exact task to the nn
        # can_move = sorted(can_move, key=lambda x: x[1])

        # don't go in path already visited, we should move this to
        # the nn some time in the future
        dont_move = 0
        for i, move in enumerate(can_move):
            if (base_i + move[2], base_j + move[3]) in self.visited_cells:
                can_move[i][1] = 0
                dont_move += 1
        self.prev_can_moves = can_move
        if debug:
            print('\b' * 85, self.counter,  "left:", max_left, "right:", max_right,
                  "up:", max_up, "down:", max_down, end='')
        if len(can_move) > dont_move:
            moves = [x[1] for x in can_move]
            output = self.network.process_input(moves)
            self.prev_choice = output
            move = max(range(len(output)), key=output.__getitem__)
            key = can_move[move][0]
            # if we choose a move with 0 distance, we are dead
            if can_move[move][1] == 0:
                if debug:
                    print(" collided")
                self.present_cell = self.dest
                self.car.killed = True
                self.calculate_score()
            else:
                if debug:
                    print(" going", "%-5s" % self.keymap[key], end='')
                self.pressed_keys[key] = True
                self.current_direction = key
                self.counter += 1
        else:
            if debug:
                print(" no possible move found")
            self.present_cell = self.dest
            self.car.killed = True
            self.calculate_score()
        if debug:
            sys.stdout.flush()
        return self.pressed_keys
