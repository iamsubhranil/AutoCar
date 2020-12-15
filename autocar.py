import pygame
import random
from carai import CarAI, get_idx
from car import Car
from roadmap import Roadmap
from math import floor

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

from config import (
    ROAD_SPRITE_LOCATION,
    BACKGROUND_SPRITE_LOCATION,
    SCALE_FACTOR_X,
    SCALE_FACTOR_Y,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    ROAD_WIDTH,
    ROAD_HEIGHT
)

def load_roads():
    images = [pygame.image.load(ROAD_SPRITE_LOCATION % i) for i in range(1, 21)]
    images = [pygame.transform.scale(x, (ROAD_WIDTH, ROAD_HEIGHT)).convert() for x in images]
    return images

def load_background():
    back = [pygame.image.load(BACKGROUND_SPRITE_LOCATION % i) for i in range(3)]
    back = [pygame.transform.scale(img, (ROAD_WIDTH, ROAD_HEIGHT)).convert() for img in back]
    return back

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

def gettile(coordinate, tiles):
    i, j = floor(coordinate[0] / ROAD_WIDTH), floor(coordinate[1] / ROAD_HEIGHT)
    if i >= SCALE_FACTOR_Y:
        i -= 1
    if j >= SCALE_FACTOR_X:
        j -= 1
    return tiles[i * SCALE_FACTOR_X + j]

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
    pygame.time.set_timer(CHECK_KEYS, 24) # polling rate

    rects = []

    pressed_keys = None

    touched_points = []
    update_tiles = []

    carai = CarAI(car, road_collection, roadmap.roadmap, path[-1])

    no_keys_pressed = {K_UP: False,
                       K_DOWN: False,
                       K_LEFT: False,
                       K_RIGHT: False}

    while running:
        pressed_keys = no_keys_pressed
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
        car.update(pressed_keys, roads)

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
