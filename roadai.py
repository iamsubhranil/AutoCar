from config import MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT

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

