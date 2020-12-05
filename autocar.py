import pygame
import random
import sys
from math import atan2, degrees, floor

from pygame.locals import (
    RLEACCEL,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT
)

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

def load_roads():
    images = [pygame.image.load("sprites/roadTexture_%03d.png" % i) for i in range(1, 21)]
    images = [pygame.transform.scale(x, (ROAD_WIDTH, ROAD_HEIGHT)).convert() for x in images]
    return images

def load_background():
    back = [pygame.image.load("sprites/background%02d.png" % i) for i in range(3)]
    back = [pygame.transform.scale(img, (ROAD_WIDTH, ROAD_HEIGHT)).convert() for img in back]
    return back

def load_roadmap():
    with open("sprites/roadmap") as f:
        roadmap = f.readlines()
        #print(roadmap)
        roadmap = filter(lambda x: len(x) > 0 and not x.startswith("#") and not x.endswith(" N\n"), roadmap)
        roadmap = [(int(x.split()[0]) - 1, set(map(int, x.split()[1:]))) for x in roadmap]
        return roadmap

def generate_possible_moves(source, coordinates):
    y0, x0 = source
    moves = []

    if y0 > coordinates[0]:
        moves.append(MOVE_LEFT)
    if y0 < coordinates[2]:
        moves.append(MOVE_RIGHT)

    if x0 > coordinates[1]:
        moves.append(MOVE_UP)
    if x0 < coordinates[3]:
        moves.append(MOVE_DOWN)

    return moves, len(moves) > 0

class RoadAI:

    def __init__(self):
        self.coordinates = None
        self.vertices = None
        self.g_scores = None
        self.f_scores = None
        self.goal = None

    def h(self, current):
        return abs(current[0] - self.goal[0]) + abs(current[1] - self.goal[1])

    def g_score(self, vertex):
        if vertex not in self.g_scores:
            return 99999 # arbitrarily large value
        return self.g_scores[vertex]

    def f_score(self, vertex):
        if vertex not in self.f_scores:
            return 99999 # arbitrarily large value
        return self.f_scores[vertex]

    def get_neighbours(self, point, coordinates):
        moves, can_move = generate_possible_moves(point, coordinates)
        nmoves = []
        for move in moves:
            nmoves.append((point[0] + move[0], point[1] + move[1]))
        #print(nmoves)
        return nmoves

    def d(self, current, pixel):
        if pixel in self.vertices:
            return 99999 # cannot be reached
        return 1

    def generate_directions(self, cameFrom, current):
        final = []
        while current in cameFrom:
            final.insert(0, cameFrom[current][1])
            current = cameFrom[current][0]
        return final

    # A* search
    # directly copied from Wikipedia
    def generate_moves(self, start):
        #print(start)
        self.f_scores = {}
        self.g_scores = {}
        #print_("Goal : " +  str(goal) + "Start:" + str(start))
        #cameFrom = {}

        self.g_scores = {}
        self.g_scores[start] = 0

        self.f_scores = {}
        self.f_scores[start] = self.h(start)

        openSet = [start]

        while len(openSet) > 0:
            current = min(openSet, key=self.f_score)
            if current == self.goal:
                #directions = self.generate_directions(cameFrom, current)
                #return directions
                return True

            openSet.remove(current)
            curr_gscore = self.g_score(current)

            neighbours = self.get_neighbours(current, self.coordinates)
            for direction, neighbour in enumerate(neighbours):
                tentative_gscore = curr_gscore + self.d(current, neighbour)
                if tentative_gscore < self.g_score(neighbour):
                    #cameFrom[neighbour] = (current, direction)
                    self.g_scores[neighbour] = tentative_gscore
                    self.f_scores[neighbour] = self.g_score(neighbour) + self.h(neighbour)
                    if neighbour not in openSet:
                        openSet.append(neighbour)
        return False

class Roadmap:

    def __init__(self):
        self.roadmap = load_roadmap()
        # mapping the displacements into directions 
        self.movement = {

            # (displacement_x, displacement_y, entry): exit
            (1, 0, None): 2,
            (1, 0, 1): 2,
            (1,-1, 1): 3,
            (1, 1, 1): 4,

            (-1, 0, None): 1,
            (-1, 0, 2): 1,
            (-1,-1, 2): 3,
            (-1, 1, 2): 4,

            (0, 1, None): 4,
            (-1, 1, 3): 1,
            (1, 1, 3): 2,
            (0, 1, 3): 4,

            (0, -1, None): 3,
            (-1, -1, 4): 1,
            (1, -1, 4): 2,
            (0, -1, 4): 3
        }

        self.exit_to_entry = {1:2, 2:1, 4:3, 3:4}
        print(self.roadmap)
        print(self.movement)
        self.ai = RoadAI()

    def norm(self, x):
        if x < -1:
            return -1
        if x > 1:
            return 1
        return x

    def get_end(self, move, start=True):
        exit = self.movement[(*move, None)]
        end_move = set([exit])
        for road in self.roadmap:
            if road[1] == end_move:
                return road[0], exit

    def inverse(self, x):
        a, b = x
        if a != 0:
            a = -a
        if b != 0:
            b = -b
        return (a, b)

    def generate_sprites(self, final_moves):
        sprites = []
        # select the first move only based on the exit
        start, exit = self.get_end(final_moves[0])
        sprites.append(start)
        #print(sprites)
        prev_move = final_moves[0]
        #entry = exit
        for move in final_moves[1:]:
            #print(prev_move, move, exit)
            entry = self.exit_to_entry[exit]
            nmove = move
            if prev_move != move:
                #prev_move = self.inverse(prev_move)
                nmove = (prev_move[0] + move[0], prev_move[1] + move[1])
            exit = self.movement[(*nmove, entry)]
            entry_exit = (entry, exit)
            while True:
                # randomly select a piece of road
                road = random.choice(self.roadmap)
                if road[1].issuperset(entry_exit):
                    sprites.append(road[0])
                    break
            prev_move = move
        sprites.append(self.get_end(self.inverse(final_moves[-1]), start=False)[0])
        return sprites

    # given a pair of source and destination vertices, it will
    # generate a random path, and return the list of sprites and
    # their coordinates which will form the path
    def generate_path(self, source, dest, coordinates):
        vertices = set()
        vertices.add(source)
        path, final_moves, result = self.generate_path_it(source, dest, coordinates)
        if result:
            return path, final_moves
        else:
            print("No path found from", source, "to", dest, "using", coordinates)
            return None, None

    def generate_path_it(self, source, dest, coordinates):
        final_path = [source]
        final_moves = []
        stack = [[source, True]]
        movestack = []
        y1, x1 = dest
        vertices = set()
        vertices.add(source)
        self.ai.coordinates = coordinates
        self.ai.vertices = vertices
        self.ai.goal = dest
        while len(stack) > 0:
            points, prepare = stack[-1]
            y0, x0 = points
            print("\b" * 80, "Checking %03d, %03d" % (y0, x0), end='')
            if prepare:
                moves, can_move = generate_possible_moves(points, coordinates)
                if not can_move:
                    #print("False1", end='')
                    stack.pop()
                    continue
                #choices = set()
                random.shuffle(moves)
                movestack.append([moves, 0])
                stack[-1][1] = False

            moves, i = movestack[-1]

            # this cannot happen anymore
            #if point != None:
            #    print("here")
            #    vertices.discard(point)
            #    final_path.pop()
            breakall = False
            for ni, move_point in enumerate(moves[i:], i):
                move_y, move_x = move_point
                #print(movestack)
                #print(points, move_y, move_x)
                point = (y0 + move_y, x0 + move_x)
                if point not in vertices:
                    #print(point[0], point[1])
                    if point[0] == y1 and point[1] == x1:
                        vertices.add(point)
                        final_path.append(dest)
                        final_moves.append(move_point)
                        #print("True0", end='')
                        return final_path, final_moves, True
                    elif self.ai.generate_moves(point):
                        vertices.add(point)
                        final_path.append(point)
                        final_moves.append(move_point)
                        movestack[-1][1] = ni + 1
                        stack.append([point, True])
                        breakall = True
                        break
                if breakall:
                    break
            if not breakall:
                movestack.pop()
                stack.pop()
        return final_path, None, False

def print_path(path, final_moves, endy, endx):
    #print(path)
    #print(final_moves)
    print()
    for j in range(endx + 1):
        for i in range(endy + 1):
            if (i, j) in path:
                try:
                    y, x = final_moves[path.index((i, j))]
                    hm = ["←", "", "→"]
                    vm = ["↑", "", "↓"]
                    print(max(hm[y + 1], vm[x + 1], key=len), end=' ')
                except:
                    print("ST", end=' ')
            else:
                print(" ", end=' ')
        print()

class Road(pygame.sprite.Sprite):

    def __init__(self, coordinate, road, roadid):
        super(Road, self).__init__()
        self.roadid = roadid
        self.surf = road
        self.surf.set_colorkey((33, 191, 143), RLEACCEL)
        self.rect = self.surf.get_rect(topleft=(coordinate[0] * ROAD_WIDTH,
                                                coordinate[1] * ROAD_HEIGHT))
        #print(self.rect)
    def update(self):
        pass

class Car(pygame.sprite.Sprite):

    CarImage = pygame.image.load("sprites/cars/hatchbackSports_01_cropped.png")
    CarImage = pygame.transform.scale(CarImage, (CAR_WIDTH, CAR_HEIGHT))

    def __init__(self):
        super(Car, self).__init__()
        self.surf = Car.CarImage.convert_alpha()
        self.rect = self.surf.get_rect(center=(CAR_WIDTH // 2, CAR_HEIGHT // 2))
        self.speed = 1
        self.maxspeed = 4
        self.movement = { K_UP: (0, -self.speed),
                         K_DOWN: (0, self.speed),
                         K_LEFT: (-self.speed, 0),
                         K_RIGHT: (self.speed, 0)}
        self.speed_right = 0.1
        self.speed_bottom = 0.1
        self.deceleration_rate = 0.90 # at each update, the speed will be this times of the previous
        self.accelaration_rate = 1.10 # at each key press, this is the amount of speed increase
        self.horizontal_move = 0
        self.vertical_move = 0
        self.target_rotation = 0
        self.current_rotation = 0
        self.rotation_delta = 8

    def norm(self, speed):
        if speed > self.maxspeed:
            return self.maxspeed
        elif speed < -self.maxspeed:
            return -self.maxspeed
        return speed

    def update(self, key_pressed, road):
        y, x = self.rect.right, self.rect.bottom

        moved_right, moved_bottom = False, False
        for key in self.movement:
            if key_pressed[key]:
                sr, sb = self.movement[key]
                if sr != 0:
                    if self.horizontal_move == 0 or \
                        (self.horizontal_move == 1 and self.speed_right < 0.01) or \
                        (self.horizontal_move == -1 and self.speed_right > -0.01):
                        delta = sr
                    else:
                        delta = abs(self.speed_right * self.accelaration_rate) * sr
                    self.speed_right += delta
                    self.speed_right = self.norm(self.speed_right)
                    self.rect.move_ip(self.speed_right, 0)
                    moved_right = True
                if sb != 0:
                    if self.vertical_move == 0 or \
                            (self.vertical_move == 1 and self.speed_bottom < 0.01) or \
                            (self.vertical_move == -1 and self.speed_bottom > 0.01):
                        delta = sb
                    else:
                        delta = abs(self.speed_bottom * self.accelaration_rate) * sb
                    self.speed_bottom += delta
                    self.speed_bottom = self.norm(self.speed_bottom)
                    self.rect.move_ip(0, self.speed_bottom)
                    moved_bottom = True

        if not moved_right:
            if self.speed_right != 0:
                self.speed_right *= self.deceleration_rate
            self.rect.move_ip(self.speed_right, 0)
            #self.speed_right = self.norm(self.speed_right)
        if not moved_bottom:
            if self.speed_bottom != 0:
                self.speed_bottom *= self.deceleration_rate
            self.rect.move_ip(0, self.speed_bottom)

        if moved_right or moved_bottom:
            self.horizontal_move, self.vertical_move = sr, sb
            newy, newx = self.rect.right, self.rect.bottom
            rad = atan2(newy - y, newx - x)
            deg = degrees(rad) - 90
            self.target_rotation = int(deg)
            self.target_rotation = self.rotation_delta * round(self.target_rotation / self.rotation_delta)
            if self.target_rotation < -180:
                self.target_rotation += 360
            elif self.target_rotation > 180:
                self.target_rotation -= 360
            cr, tr = self.current_rotation, self.target_rotation
            if (cr >= 0 and tr >= 0) or (cr < 0 and tr < 0):
                if cr > tr:
                    #pass
                    self.rotation_delta = -abs(self.rotation_delta)
                else:
                    #pass
                    self.rotation_delta = abs(self.rotation_delta)
            else:
                if cr >= 0:
                    positive_rotation = 180 - cr + 180 + tr
                    negative_rotation = cr + abs(tr)
                else:
                    positive_rotation = tr + abs(cr)
                    negative_rotation = (180 + cr) + (180 - tr)
                #print(positive_rotation, negative_rotation)
                if positive_rotation > negative_rotation:
                    self.rotation_delta = -abs(self.rotation_delta)
                else:
                    self.rotation_delta = abs(self.rotation_delta)


        #print(self.target_rotation, self.current_rotation, self.rotation_delta)

        if self.current_rotation != self.target_rotation:
            #bl, br = self.rect.bottomleft, self.rect.bottomright
            center = self.rect.center
            self.surf = pygame.transform.rotate(Car.CarImage, self.current_rotation).convert_alpha()
            #self.rect.bottomleft, self.rect.bottomright = bl, br
            self.rect.center = center
            self.current_rotation += self.rotation_delta
            if self.current_rotation == 180 and self.target_rotation == -180 \
                    or self.current_rotation == -180 and self.target_rotation == 180:
                self.current_rotation = self.target_rotation
            elif self.current_rotation > 180:
                self.current_rotation -= 360
            elif self.current_rotation < -180:
                self.current_rotation += 360
                #self.rotation_delta = abs(self.rotation_delta)
            #self.current_rotation = self.current_rotation % 180

        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT
        if pygame.sprite.spritecollideany(self, road):
            return
        else:
            self.rect.right, self.rect.bottom = y, x

def gettile(coordinate, tiles):
    i, j = floor(coordinate[0] / ROAD_WIDTH), floor(coordinate[1] / ROAD_HEIGHT)
    if i >= SCALE_FACTOR_Y:
        i -= 1
    if j >= SCALE_FACTOR_X:
        j -= 1
    return tiles[i * SCALE_FACTOR_X + j]

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
        self.current_direction = None
        self.goal_cell = None
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
        self.calculate_first_move(0, 0)

    def should_go(self, y, x, base_i, base_j):
        if (base_i + y, base_j + x) in self.visited_cells:
            return False
        return True

    def calculate_first_move(self, base_i, base_j):
        # calculate max distance on the right
        max_right = SCALE_FACTOR_Y - 1
        direction = self.entering_from[K_LEFT]
        for i in range(0, SCALE_FACTOR_Y):
            road = self.road_collection[(base_i + i) * SCALE_FACTOR_X + base_j]
            if road == None or (direction not in self.roadmap[road.roadid]):
                print("right:", i, direction, self.roadmap[road.roadid])
                max_right = i
                break
        max_down = SCALE_FACTOR_X - 1
        base_idx = base_i * SCALE_FACTOR_X
        direction = self.entering_from[K_UP]
        for i in range(0, SCALE_FACTOR_X):
            road = self.road_collection[base_idx + base_j + i]
            if road == None or (direction not in self.roadmap[road.roadid]):
                print("down:", i, direction, self.roadmap[road.roadid])
                max_down = i
                break
        print(max_right, max_down)
        if max_right > max_down:
            self.pressed_keys[K_RIGHT] = True
            self.goal_cell = (base_i + max_right, base_j)
            self.current_direction = K_RIGHT
        else:
            self.pressed_keys[K_DOWN] = True
            self.goal_cell = (base_i, base_j + max_down)
            self.current_direction = K_DOWN
        return self.pressed_keys

    def calculate_next_move(self):
        for key in self.pressed_keys:
            self.pressed_keys[key] = False
        base_i, base_j = get_idx(*self.car.rect.center)
        if self.goal_cell != None:
            if (base_i, base_j) == self.goal_cell:
                if self.goal_cell == self.dest:
                    return self.pressed_keys
                self.goal_cell = None
            else:
               self.pressed_keys[self.current_direction] = True
               self.visited_cells.add((base_i, base_j))
            return self.pressed_keys
        moving_right, moving_bottom = self.car.horizontal_move, self.car.vertical_move
        if moving_bottom != 0:
            # we're moving up or down
            max_front = 0
            base_idx = base_i * SCALE_FACTOR_X
            # if moving_bottom == +1, moving_bottom + 1 == 2, // 2 works
            # if moving_bottom == -1, moving_bottom + 1 == 0, hence -1 works
            for end in range(base_j, ((SCALE_FACTOR_X + 1) * (moving_bottom + 1) // 2) - 1 , moving_bottom):
                if self.road_collection[base_idx + end] == None:
                    max_front = abs(base_j - end)
                    break
            if max_front > 0:
                max_front -= 1
            max_right = SCALE_FACTOR_Y - base_i - 1
            # make sure we take right turn on valid roads
            direction = self.entering_from[K_RIGHT]
            for end in range(base_i + 1, SCALE_FACTOR_Y, 1):
                road = self.road_collection[end * SCALE_FACTOR_X + base_j]
                if road == None or (direction not in self.roadmap[road.roadid]):
                    max_right = abs(base_i - end) - 1
                    break
            max_left = base_i
            direction = self.entering_from[K_LEFT]
            for end in range(base_i - 1, -1, -1):
                road = self.road_collection[end * SCALE_FACTOR_X + base_j]
                if road == None or (direction not in self.roadmap[road.roadid]):
                    max_left = abs(base_i - end) - 1
                    break

            if (max_right > max_left and self.should_go(max_right, 0, base_i, base_j)) or \
                    not self.should_go(-max_left, 0, base_i, base_j):
                key = K_RIGHT
                delta = max_right
            else:
                key = K_LEFT
                delta = -max_left

            self.pressed_keys[key] = True
            self.current_direction = key
            self.goal_cell = (base_i + delta, base_j)
            print("front:", max_front, "right:", max_right, "left:", max_left, self.counter)
        elif moving_right != 0:
            # we're moving left or right
            max_front = 0
            base_idx = base_i
            for end in range(base_idx, ((SCALE_FACTOR_Y + 1) * (moving_right + 1) // 2) - 1, moving_right):
                if self.road_collection[end * SCALE_FACTOR_X + base_j] == None:
                    max_front = abs(base_i - end)
                    break
            if max_front > 0:
                max_front -= 1
            base_idx = base_i * SCALE_FACTOR_X
            max_up = base_j
            direction = self.entering_from[K_UP]
            for end in range(base_j - 1, -1, -1):
                road = self.road_collection[base_idx + end]
                if road == None or (direction not in self.roadmap[road.roadid]):
                    max_up = abs(base_j - end) - 1
                    break
            max_down = SCALE_FACTOR_X - base_j - 1
            direction = self.entering_from[K_DOWN]
            for end in range(base_j + 1, SCALE_FACTOR_X, 1):
                road = self.road_collection[base_idx + end]
                if road == None or (direction not in self.roadmap[road.roadid]):
                    max_down = abs(base_j - end) - 1
                    break

            if max_up > max_down and self.should_go(0, -max_up, base_i, base_j) or \
                    not self.should_go(0, max_down, base_i, base_j):
                key = K_UP
                delta = -max_up
            else:
                key = K_DOWN
                delta = max_down

            self.pressed_keys[key] = True
            self.current_direction = key
            self.goal_cell = (base_i, base_j + delta)
            print("front:", max_front, "up:", max_up, "down:", max_down, self.counter)
        self.counter += 1
        return self.pressed_keys

def main():
    pygame.init()

    roadmap = Roadmap()
    endy, endx = SCALE_FACTOR_Y, SCALE_FACTOR_X
    path, final_moves = roadmap.generate_path((0, 0), (endy - 1, endx - 1), (0, 0, endy - 1, endx - 1))
    print_path(path, final_moves, endy, endx)
    sprite_indices = roadmap.generate_sprites(final_moves)

    roads = pygame.sprite.Group()
    #print(len(path), len(final_moves), len(sprite_indices))

    screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
    road_images = load_roads()
    road_collection = [None] * (SCALE_FACTOR_X * SCALE_FACTOR_Y)
    for i, coordinate in enumerate(path):
        r = Road(coordinate, road_images[sprite_indices[i]], sprite_indices[i])
        roads.add(r)
        road_collection[coordinate[0] * SCALE_FACTOR_X + coordinate[1]] = r
    background_group = pygame.sprite.Group()
    background_collection = []
    back = load_background()
    for i in range(endy):
        for j in range(endx):
            #if (i, j) not in path:
            bg = Road((i, j), random.choice(back), 0)
            background_group.add(bg)
            background_collection.append(bg)
    running = True

    clock = pygame.time.Clock()

    car = Car()

    for back in background_group:
        screen.blit(back.surf, back.rect)
    for road in roads:
        screen.blit(road.surf, road.rect)
    screen.blit(car.surf, car.rect)
    pygame.display.update()

    update_tiles = [None] * 9
    update_tiles[8] = car

    CHECK_KEYS = pygame.USEREVENT + 1
    pygame.time.set_timer(CHECK_KEYS, 16) # polling rate

    rects = []

    pressed_keys = None

    touched_points = []
    update_tiles = []

    carai = CarAI(car, road_collection, roadmap.roadmap, path[-1])

    while running:
        if pygame.event.peek():
            for event in pygame.event.get():
                if event == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False
                elif event.type == CHECK_KEYS:
                    pressed_keys = carai.calculate_next_move()

        # calculate indices of the surrounding tiles of the car
        #   A   B   C
        #   D  CAR  E
        #   F   G   H
        # not all times all of the surrounding tiles will be available,
        # that is also considered
        cary, carx = car.rect.center
        base_i, base_j = get_idx(cary, carx)
        idxs = [(base_i * SCALE_FACTOR_X) + base_j]
        if base_i > 0:
            new_i = (base_i - 1) * SCALE_FACTOR_X
            idxs.append(new_i + base_j)
            if base_j > 0:
                idxs.append(new_i + base_j - 1)
            if base_j < SCALE_FACTOR_X - 1:
                idxs.append(new_i + base_j + 1)
        if base_i < SCALE_FACTOR_Y - 1:
            new_i = (base_i + 1) * SCALE_FACTOR_X
            idxs.append(new_i + base_j)
            if base_j > 0:
                idxs.append(new_i + base_j - 1)
            if base_j < SCALE_FACTOR_X - 1:
                idxs.append(new_i + base_j + 1)
        if base_j > 0:
            idxs.append((base_i * SCALE_FACTOR_X) + (base_j - 1))
        if base_j < SCALE_FACTOR_X - 1:
            idxs.append((base_i * SCALE_FACTOR_X) + (base_j + 1))

        # get the tiles
        update_tiles = []
        for i in idxs:
            update_tiles.append(background_collection[i])
        for i in idxs:
            if road_collection[i] != None:
                update_tiles.append(road_collection[i])

        rects = [(v.surf, v.rect) for v in update_tiles]
        #print([v[1] for v in rects])
        if pressed_keys != None:
            car.update(pressed_keys, roads)
            pressed_keys = None

        screen.blits(rects)
        screen.blit(car.surf, car.rect)

        pygame.display.update([v[1] for v in rects])
        pygame.display.update(car.rect)
        #pygame.display.flip()
        #print(clock.get_rawtime(), clock.get_fps())
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
