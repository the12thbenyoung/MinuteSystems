from graphics import *
import numpy as np

class Tube:
    def __init__(self, x, y, edgeLength, win):
        self.x = x
        self.y = y

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

    def showAsPresent(self):
        self.circle.setFill('black')

    def showAsPicked(self):
        self.circle.setFill('red')

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



if __name__ == '__main__':
    EDGE_LENGTH = 30
    NUM_RACKS = 5
    TUBES_ALONG_X = 8
    TUBES_ALONG_Y = 12
    windowX= EDGE_LENGTH*(NUM_RACKS*(TUBES_ALONG_X+1)-1)
    windowY = EDGE_LENGTH*(TUBES_ALONG_Y)
    win = GraphWin('tubes', windowX, windowY) # give title and dimensions
    initializeTray(NUM_RACKS, EDGE_LENGTH, TUBES_ALONG_X, TUBES_ALONG_Y, win)
    win.getMouse()

