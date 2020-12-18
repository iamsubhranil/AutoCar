import pygame
from config import (
    CAR_WIDTH,
    CAR_HEIGHT,
    CAR_SPRITE_LOCATION,
    CAR_SPRITE_COUNT,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,

    CAR_MAXSPEED,
    CAR_ACCELERATION_PERCEN,
    CAR_DECELERATION_PERCEN,
    CAR_ROTATION_DELTA,
)

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
)

from math import atan2, degrees
import random

class Car(pygame.sprite.Sprite):

    CarImages = [pygame.image.load(CAR_SPRITE_LOCATION % i) for i in range(CAR_SPRITE_COUNT)]
    CarImages = [pygame.transform.scale(image, (CAR_WIDTH, CAR_HEIGHT)) for image in CarImages]

    def __init__(self):
        super(Car, self).__init__()
        self.img = random.choice(Car.CarImages)
        self.surf = self.img.convert_alpha()
        self.rect = self.surf.get_rect(center=(CAR_WIDTH // 2, CAR_HEIGHT // 2))
        self.speed = 1
        self.maxspeed = CAR_MAXSPEED
        self.movement = { K_UP: (0, -self.speed),
                         K_DOWN: (0, self.speed),
                         K_LEFT: (-self.speed, 0),
                         K_RIGHT: (self.speed, 0)}
        self.speed_right = 0.1
        self.speed_bottom = 0.1
        self.deceleration_rate = CAR_DECELERATION_PERCEN # at each update, the speed will be this times of the previous
        self.accelaration_rate = CAR_ACCELERATION_PERCEN # at each key press, this is the amount of speed increase
        self.horizontal_move = 0
        self.vertical_move = 0
        self.target_rotation = 0
        self.current_rotation = 0
        self.rotation_delta = CAR_ROTATION_DELTA
        self.killed = False

    def norm(self, speed):
        if speed > self.maxspeed:
            return self.maxspeed
        elif speed < -self.maxspeed:
            return -self.maxspeed
        return speed

    def update(self, key_pressed, road):
        if self.killed:
            return
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
            self.surf = pygame.transform.rotate(self.img, self.current_rotation).convert_alpha()
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
            self.killed = True
            self.rect.right, self.rect.bottom = y, x

