from graphics import *
import numpy as np
from makeTestInput import writeInput
from PIL import Image

#tube statuses
ABSENT = 0
PRESENT = 1
TARGET = 2
PICKED = 3

class trayStatusViewer:
    #represents one tube - one square/circle displayed in the window
    class Tube:
        def __init__(self, x, y, edgeLength, win):
            self.x = x
            self.y = y
            self.status = ABSENT

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

        def showAsTarget(self):
            self.circle.setFill('red')
            self.status = TARGET

        def showAsPicked(self):
            self.circle.setFill(color_rgb(52, 217, 90))
            self.status = PICKED

        def getStatus(self):
            return self.status

    def __init__(self, edgeLength, numRacks, tubesAlongX, tubesAlongY):
        self.EDGE_LENGTH = edgeLength
        self.NUM_RACKS = numRacks
        self.TUBES_ALONG_X = tubesAlongX
        self.TUBES_ALONG_Y = tubesAlongY

        windowX = self.EDGE_LENGTH*(self.NUM_RACKS*(self.TUBES_ALONG_X+1)-1)
        windowY = self.EDGE_LENGTH*(self.TUBES_ALONG_Y+3)
        self.win = GraphWin('tubes', windowX, windowY) # give title and dimensions

        #initialize tray of ABSENT tubes
        #3d list to hold tube objects - rack, x-coordinate, y-coordinate
        tubes = []
        for rack in range(self.NUM_RACKS):
            x_list = []
            for x in range(self.TUBES_ALONG_X):
                y_list = []
                for y in range(self.TUBES_ALONG_Y):
                    x_coor = self.EDGE_LENGTH*(rack*(self.TUBES_ALONG_X+1) + x)
                    y_coor = self.EDGE_LENGTH*(y+1)
                    y_list.append(self.Tube(x_coor, y_coor, self.EDGE_LENGTH, self.win))
                x_list.append(y_list)
            tubes.append(x_list)
        self.tubes = tubes

        #draw column indices between each pair of trays
        for rack in range(1, self.NUM_RACKS):
            for row in range(self.TUBES_ALONG_Y):
                x_coor = int(np.round(self.EDGE_LENGTH*(rack*(self.TUBES_ALONG_X+1)-0.5)))
                y_coor = int(np.round(self.EDGE_LENGTH*(row + 1.5)))
                Text(Point(x_coor, y_coor), str(TUBES_ALONG_Y-row)).draw(self.win)

        #draw row indices above and below each tray
        alphabet = 'ABCDEFGH'
        top_y_coor = self.EDGE_LENGTH*(self.TUBES_ALONG_Y+1.5)
        bottom_y_coor = self.EDGE_LENGTH*0.5
        for rack in range(self.NUM_RACKS):
            for col in range(self.TUBES_ALONG_X):
                x_coor = self.EDGE_LENGTH*(rack*(self.TUBES_ALONG_X+1) + col + 0.5)
                Text(Point(x_coor, top_y_coor), alphabet[col]).draw(self.win)

                Text(Point(x_coor, bottom_y_coor), alphabet[col]).draw(self.win)

        #draw tray numbers beneath trays
        y_coor = self.EDGE_LENGTH*(TUBES_ALONG_Y+2.3)
        for rack in range(self.NUM_RACKS):
            x_coor = self.EDGE_LENGTH*((self.TUBES_ALONG_X+1)*(rack + 0.5)-0.5)
            rackNum = Text(Point(x_coor, y_coor), rack+1)
            rackNum.setSize(20)
            rackNum.draw(self.win)

        # self.win.getMouse()

    #read from files to determine which tubes start as ABSENT, PRESENT, or TARGET and color accordingly
    def newTray(self, locationsFilename, targetsFilename):
        #reset all tubes to ABSENT
        for rack in self.tubes:
            for col in rack:
                for tube in col:
                    tube.showAsAbsent()

        #set tubes listed in file as PRESENT or ABSENT
        with open(locationsFilename, 'r') as file:
            for line in file.readlines():
                row = list(map(lambda x: int(x)-1, line.split(',')))
                #make sure all three are in bounds
                if all([0 <= m and m < bound for m, bound in zip(row, [self.NUM_RACKS, self.TUBES_ALONG_X, self.TUBES_ALONG_Y])]):
                    rack, x, y = row
                    if self.tubes[rack][x][y].getStatus() != PRESENT:
                        self.tubes[rack][x][y].showAsPresent()

        #set some PRESENT tubes as TARGETs
        with open(targetsFilename, 'r') as file:
            for line in file.readlines():
                row = list(map(lambda x: int(x)-1, line.split(',')))
                #make sure all three are in bounds
                if all([0 <= m and m < bound for m, bound in zip(row, [self.NUM_RACKS, self.TUBES_ALONG_X, self.TUBES_ALONG_Y])]):
                    rack, x, y = row
                    #only set PRESENT tubes to TARGET
                    if self.tubes[rack][x][y].getStatus() == PRESENT:
                        self.tubes[rack][x][y].showAsTarget()
                    # else:
                    #     raise Warning("Target tube at rack {}, position {},{} not recorded as present in tray!".format(rack, x, y))


    def pickTube(self, rack, x, y):
        if all([0 <= m and m < bound for m, bound in zip([rack, x, y], [self.NUM_RACKS, self.TUBES_ALONG_X, self.TUBES_ALONG_Y])]):
            target = self.tubes[rack][x][y]
            if target.getStatus() == TARGET:
                target.showAsPicked()
            elif target.getStatus() == ABSENT:
                print("Tube is not in rack!")
            elif target.getStatus() == PRESENT:
                print("Tube was not supposed to be picked!")
            elif target.getStatus() == PICKED:
                print("Tube has already been picked!")
        else:
            print("Tube out of bounds!")

if __name__ == '__main__':
    NUM_RACKS = 6
    EDGE_LENGTH = 25 if NUM_RACKS == 6 else 30
    TUBES_ALONG_X = 8
    TUBES_ALONG_Y = 12

    #edgeLength, numRacks, tubesAlongX, tubesAlongY
    viewer = trayStatusViewer(EDGE_LENGTH, NUM_RACKS, TUBES_ALONG_X, TUBES_ALONG_Y)

    #generate random present/absent sample input
    writeInput(300, NUM_RACKS, TUBES_ALONG_X, TUBES_ALONG_Y, 'present_input.csv')
    writeInput(50, NUM_RACKS, TUBES_ALONG_X, TUBES_ALONG_Y, 'target_input.csv')

    viewer.newTray('present_input.csv', 'target_input.csv')

    target = input('Enter coordinates <rack>,<x>,<y>: ')
    while target and ',' in target:
        rack, x, y = map(lambda x: int(x)-1, target.split(','))
        viewer.pickTube(rack, x, y)
        target = input('Enter coordinates <rack>,<x>,<y>: ')

    viewer.win.postscript(file='img.eps', colormode='color')
    img = Image.open('img.eps')
    img.save('tray.jpg', 'jpeg')


