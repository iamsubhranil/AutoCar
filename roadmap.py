from config import ROADMAP_LOCATION
from roadai import RoadAI, generate_possible_moves
import random

def load_roadmap():
    with open(ROADMAP_LOCATION) as f:
        roadmap = f.readlines()
        #print(roadmap)
        roadmap = filter(lambda x: len(x) > 0 and not x.startswith("#") and not x.endswith(" N\n"), roadmap)
        roadmap = [(int(x.split()[0]) - 1, set(map(int, x.split()[1:]))) for x in roadmap]
        return roadmap

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
