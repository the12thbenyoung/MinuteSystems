from graphics import *
import numpy as np

EDGE_LENGTH = 30
NUM_RACKS = 5
TUBES_ALONG_X = 8
TUBES_ALONG_Y = 12

#tube statuses
ABSENT = 0
PRESENT = 1
PICKED = 2

class Tube:
    def __init__(self, x, y, edgeLength, win):
        self.x = x
        self.y = y
        self.status = 0

        square = Rectangle(Point(x, y), Point(x + edgeLength, y + edgeLength))
        square.setOutline('black')
        square.setFill('white')
        self.square = square

        circle = Circle(Point(np.round(x + edgeLength/2), (np.round(y + edgeLength/2))), np.round(edgeLength*0.4))
        self.circle = circle
        circle.setOutline('black')
        circle.setFill('white')

        square.draw(win)
        circle.draw(win)

    def showAsAbsent(self):
        self.circle.setFill('white')
        self.status = ABSENT

    def showAsPresent(self):
        self.circle.setFill('black')
        self.status = PRESENT

    def showAsPicked(self):
        self.circle.setFill('red')
        self.status = PICKED

    def getStatus(self):
        return self.status


def initializeTray(numRacks, edgeLength, tubesAlongX, tubesAlongY, win):
    #3d list to hold tube objects - rack, x-coordinate, y-coordinate
    tubes = []
    for rack in range(numRacks):
        x_list = []
        for x in range(tubesAlongX):
            y_list = []
            for y in range(tubesAlongY):
                y_list.append(Tube(edgeLength*(rack*(tubesAlongX+1) + x), edgeLength*y, edgeLength, win))
            x_list.append(y_list)
        tubes.append(x_list)
    return tubes

def newTray(filename, tubes):
    #reset all tubes to empty
    for rack in tubes:
        for col in rack:
            for tube in col:
                tube.showAsAbsent()

    #set tubes listed in file as present
    with open(filename, 'r') as file:
        for line in file.readlines():
            row = list(map(lambda x: int(x)-1, line.split(',')))
            #make sure all three are in bounds
            if all([0 <= m and m < bound for m, bound in zip(row, [NUM_RACKS, TUBES_ALONG_X, TUBES_ALONG_Y])]):
                rack, x, y = row
                tubes[rack][x][y].showAsPresent()

def removeTube(tubes, rack, x, y):
    if all([0 <= m and m < bound for m, bound in zip([rack, x, y], [NUM_RACKS, TUBES_ALONG_X, TUBES_ALONG_Y])]):
        target = tubes[rack][x][y]
        if target.getStatus() == PRESENT:
            target.showAsPicked()
        elif target.getStatus() == ABSENT:
            print("Tube is not in rack!")
        elif target.getStatus() == PICKED:
            print("Tube has already been picked!")
    else:
        print("Tube out of bounds!")

def userRemoveTubes(tubes):
    target = input('Enter coordinates <rack>,<x>,<y>: ')
    while target and ',' in target:
        rack, x, y = map(lambda x: int(x)-1, target.split(','))
        removeTube(tubes, rack, x, y)
        target = input('Enter coordinates <rack>,<x>,<y>: ')

if __name__ == '__main__':
    windowX= EDGE_LENGTH*(NUM_RACKS*(TUBES_ALONG_X+1)-1)
    windowY = EDGE_LENGTH*(TUBES_ALONG_Y)
    win = GraphWin('tubes', windowX, windowY) # give title and dimensions
    tubes = initializeTray(NUM_RACKS, EDGE_LENGTH, TUBES_ALONG_X, TUBES_ALONG_Y, win)
    win.getMouse()
    newTray('testInput.csv', tubes)
    userRemoveTubes(tubes)
    win.getMouse()

