import sys
import os

SCREEN_WIDTH = 720
SCREEN_HEIGHT = 720

ROAD_WIDTH = 48
ROAD_HEIGHT = 48

CAR_HEIGHT = 40
CAR_WIDTH = 40

SCALE_FACTOR_Y = SCREEN_WIDTH // ROAD_WIDTH
SCALE_FACTOR_X = SCREEN_HEIGHT // ROAD_HEIGHT

MOVE_LEFT = (-1, 0)
MOVE_RIGHT = (1, 0)
MOVE_UP = (0, -1)
MOVE_DOWN = (0, 1)

CAR_SPRITE_LOCATION = "sprites/cars/car%d.png"
CAR_SPRITE_COUNT = 7

ROADMAP_LOCATION = "sprites/roadmap"
ROAD_SPRITE_LOCATION = "sprites/roadTexture_%03d.png"
ROAD_SPRITE_OFFSET = 1
ROAD_SPRITE_COUNT = 20
ROAD_COLORKEY = (33, 191, 143)

BACKGROUND_SPRITE_LOCATION = "sprites/background%02d.png"
BACKGROUND_SPRITE_COUNT = 3

INNER_LAYERS = [8]
AGENT_COUNT = 200

KEYPRESS_INTERVAL_MS = 24
FLUSH_INTERVAL_MS = 200
TARGET_FPS = 60

CAR_ACCELERATION_PERCEN = 1.10
CAR_DECELERATION_PERCEN = 0.90
CAR_ROTATION_DELTA = 8
CAR_MAXSPEED = 4

CROSSOVER_GENE_COUNT = 0
CARRYOVER_PARENT_COUNT = 2
USE_UNIFORM_CROSSOVER = False
MUTATION_PROBABILITY = 0.1
PARENT_SELECTION_COUNT = 5
USE_UNIQUE_PARENTS = True

AGENT_ALIVE_WEIGHT = 0.5


def default(val):
    return " (default: " + str(val) + ")"


def argerr(arg, *err):
    print("AutoCar: error: argument", "-" + arg[0] + "/--" + arg + ":", *err)
    print("Use -h to see argument help.")
    sys.exit(0)


def verify_lowerbound(values, args, bound):
    for arg in args:
        v = values.__dict__[arg]
        if v and v < bound:
            argerr(arg, "expected a value >= " + str(bound) + "!")


def verify_bothbound(values, args, lower, upper):
    for arg in args:
        v = values.__dict__[arg]
        if v and (lower > v or v > upper):
            argerr(arg, "expected " + str(lower) +
                   " <= value <= " + str(upper) + "!")


def parse_args():
    global INNER_LAYERS
    global AGENT_COUNT
    global KEYPRESS_INTERVAL_MS
    global FLUSH_INTERVAL_MS
    global TARGET_FPS
    global CAR_ACCELERATION_PERCEN
    global CAR_DECELERATION_PERCEN
    global CAR_ROTATION_DELTA
    global CAR_MAXSPEED
    global CROSSOVER_GENE_COUNT
    global CARRYOVER_PARENT_COUNT
    global USE_UNIFORM_CROSSOVER
    global MUTATION_PROBABILITY
    global PARENT_SELECTION_COUNT
    global USE_UNIQUE_PARENTS
    global AGENT_ALIVE_WEIGHT

    from argparse import ArgumentParser

    parser = ArgumentParser(prog="AutoCar",
                            usage="python autocar.py [args..]",
                            description="A program to automatically train some car agents"
                            + " using neural networks and a genetic algorithm.")

    parser.add_argument("-n", "--numagents", metavar="N", help="number of agents in each generation"
                        + default(AGENT_COUNT), type=int)
    parser.add_argument("-l", "--layers", metavar="L1 L2", help="dimension of inner layers of the neural network"
                        + default(INNER_LAYERS), type=int, nargs='*')
    parser.add_argument("-k", "--keypress", metavar="K",
                        help="interval between consecutive keypresses, in ms"
                        + default(KEYPRESS_INTERVAL_MS), type=int)
    parser.add_argument("-o", "--output", metavar="O",
                        help="interval between consecutive output flushes, in ms"
                        + default(FLUSH_INTERVAL_MS), type=int)
    parser.add_argument("-f", "--fps", metavar="F",
                        help="target fps" + default(TARGET_FPS), type=int)

    parser.add_argument("-a", "--accel", metavar="A", help="acceleration percentage of the car"
                        + default(CAR_ACCELERATION_PERCEN), type=float)
    parser.add_argument("-d", "--decel", metavar="D", help="deceleration percentage of the car"
                        + default(CAR_DECELERATION_PERCEN), type=float)
    parser.add_argument("-r", "--rotation", metavar="R", help="rotation delta of the car, in degrees"
                        + default(CAR_ROTATION_DELTA), type=int)
    parser.add_argument("-s", "--speed", metavar="S", help="maximum speed of the car"
                        + default(CAR_MAXSPEED), type=int)

    parser.add_argument("-g", "--genes", metavar="G", help="number of parent genes participating in a crossover"
                        + default(CROSSOVER_GENE_COUNT or "all"), type=int)
    parser.add_argument("-c", "--carryover", metavar="C", help="number of parents carried over to the next generation as children"
                        + default(CARRYOVER_PARENT_COUNT), type=int)
    parser.add_argument("-u", "--uniform", help="give all parents uniform importance while sampling for crossover"
                        + default(USE_UNIFORM_CROSSOVER), action="store_true")
    parser.add_argument("-m", "--mutation", metavar="M", help="mutation probability for each child in range [0, 1.0]"
                        + default(MUTATION_PROBABILITY), type=float)
    parser.add_argument("-p", "--parent", metavar="P", help="number of selected parents"
                        + default(PARENT_SELECTION_COUNT), type=int)
    parser.add_argument("-q", "--unique", help="select parents based on unique scores"
                        + default(USE_UNIQUE_PARENTS), action="store_true")

    parser.add_argument("-w", "--weight", metavar="W", help="weight given to the age of an agent in range [0, 1.0],"
                        + " the rest is given to its correctness" +
                        default(AGENT_ALIVE_WEIGHT),
                        type=float)
    values = parser.parse_args()

    verify_lowerbound(values, ["numagents", "keypress", "output", "fps",
                               "rotation", "speed", "genes", "carryover",
                               "parent"], 1)

    verify_lowerbound(values, ["accel", "decel"], 0.0)

    verify_bothbound(values, ["mutation", "weight"], 0.0, 1.0)

    if values.layers:
        for x in values.layers:
            if x < 1:
                argerr("layers", "all layers of the network must contain > 0 nodes!")
        INNER_LAYERS = values.layers

    if values.keypress:
        KEYPRESS_INTERVAL_MS = values.keypress
    if values.output:
        FLUSH_INTERVAL_MS = values.output
    if values.fps:
        TARGET_FPS = values.fps
    if values.rotation:
        CAR_ROTATION_DELTA = values.rotation
    if values.speed:
        CAR_MAXSPEED = values.speed

    if values.accel:
        if values.accel < 1.0:
            print("warning: an acceleration value of " +
                  str(values.accel) + " will actually decelerate the car!")
        CAR_ACCELERATION_PERCEN = values.accel

    if values.decel:
        if values.decel > 1.0:
            print("warning: a deceleration value of " +
                  str(values.decel) + " will actually accelerate the car!")
        CAR_DECELERATION_PERCEN = values.decel

    if values.numagents:
        AGENT_COUNT = values.numagents

    if values.parent:
        PARENT_SELECTION_COUNT = values.parent

    if PARENT_SELECTION_COUNT > AGENT_COUNT:
        argerr("parent", "number of selected parents(" + str(PARENT_SELECTION_COUNT) + ")"
               + " must not be greater than total agents(" + str(AGENT_COUNT) + ")!")

    if values.genes:
        CROSSOVER_GENE_COUNT = values.genes

    if CROSSOVER_GENE_COUNT > PARENT_SELECTION_COUNT:
        argerr("genes", "number of participating genes(" + str(CROSSOVER_GENE_COUNT) + ")"
               + " cannot be greater than selected parents(" + str(PARENT_SELECTION_COUNT) + ")!")

    if values.carryover:
        CARRYOVER_PARENT_COUNT = values.carryover

    if CARRYOVER_PARENT_COUNT > PARENT_SELECTION_COUNT:
        argerr("carryover", "number of carried over parents (" + str(CARRYOVER_PARENT_COUNT) +
               ") cannot be greater than selected parents(" + str(PARENT_SELECTION_COUNT) + ")!")

    if values.mutation:
        MUTATION_PROBABILITY = values.mutation
    if values.weight:
        AGENT_ALIVE_WEIGHT = values.weight

    USE_UNIQUE_PARENTS = values.unique
    USE_UNIFORM_CROSSOVER = values.uniform

    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


parse_args()
