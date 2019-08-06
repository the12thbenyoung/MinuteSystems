import numpy as np
from gui.makeTestInput import writeInput
from PIL import Image, ImageDraw, ImageFont

#tube statuses
ABSENT = 0
PRESENT = 1
TARGET = 2
PICKED = 3

#font to use for index labels
indexFont = ImageFont.truetype('COURIER.TTF', size=15)
#bigger font to use for rack labels
rackFont = ImageFont.truetype('COURIER.TTF', size=25)

alphabet = 'ABCDEFGH'
lowerAlphabet = 'abcdefgh'

class trayStatusViewer:
    #represents one tube - one square/circle displayed in the window
    class Tube:

        def __init__(self, x, y, edgeLength, draw):
            self.x = x
            self.y = y
            self.status = ABSENT
            self.draw = draw
            self.edgeLength = edgeLength

            #draw empty rectangle and circle to represent tube
            draw.rectangle([(x, y), \
                            (x + edgeLength, y + edgeLength)], \
                           fill=(255,255,255), \
                           outline=(0,0,0))
            draw.ellipse([(x + edgeLength/10, y + edgeLength/10), \
                          (x + edgeLength*9/10, y + edgeLength*9/10)], \
                           fill=(255,255,255), \
                           outline=(0,0,0))

        #draw the circle representing the tube filled with given color
        #and black outline
        def __colorTube(self, color):
            self.draw.ellipse([(self.x + self.edgeLength/10, self.y + self.edgeLength/10), \
                               (self.x + self.edgeLength*9/10, self.y + self.edgeLength*9/10)], \
                              fill=color, \
                              outline=(0,0,0))

        def showAsAbsent(self):
            #color tube white
            self.__colorTube((255,255,255))
            self.status = ABSENT

        def showAsPresent(self):
            #color tube black
            self.__colorTube((0,0,0))
            self.status = PRESENT

        def showAsTarget(self):
            #color tube red
            self.__colorTube((255,0,0))
            self.status = TARGET

        def showAsPicked(self):
            #color tube green
            self.__colorTube((52, 217, 90))
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
        #image that represents the whole tray
        self.win = Image.new(mode='RGB', size=(windowX,windowY), color=(255,255,255))
        #object used to draw shapes on win
        self.draw = ImageDraw.Draw(self.win)

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
                    y_list.append(self.Tube(x_coor, y_coor, self.EDGE_LENGTH, self.draw))
                x_list.append(y_list)
            tubes.append(x_list)
        self.tubes = tubes

        #draw column indices between each pair of trays
        for rack in range(1, self.NUM_RACKS):
            x_coor = int(np.round(self.EDGE_LENGTH*(rack*(self.TUBES_ALONG_X+1)-0.85)))
            single_digit_x_coor = x_coor + int(np.round(self.EDGE_LENGTH*0.18))
            for row in range(self.TUBES_ALONG_Y):
                #if row is a singe digit, shift the text a bit to the right
                mod_x_coor = single_digit_x_coor if TUBES_ALONG_Y-row < 10 else x_coor
                y_coor = int(np.round(self.EDGE_LENGTH*(row + 1.27)))
                self.draw.text((mod_x_coor,y_coor),
                               text = str(TUBES_ALONG_Y-row),
                               font = indexFont,
                               fill = (0,0,0))

        #draw row indices above and below each rack
        bottom_y_coor = self.EDGE_LENGTH*(self.TUBES_ALONG_Y+1.2)
        top_y_coor = self.EDGE_LENGTH*0.45
        for rack in range(self.NUM_RACKS):
            for col in range(self.TUBES_ALONG_X):
                x_coor = self.EDGE_LENGTH*(rack*(self.TUBES_ALONG_X+1) + col + 0.32)
                self.draw.text((x_coor, bottom_y_coor),
                               text = alphabet[col],
                               font = indexFont,
                               fill=(0,0,0))
                self.draw.text((x_coor, top_y_coor),
                               text = alphabet[col],
                               font = indexFont,
                               fill=(0,0,0))

        #draw rack numbers beneath racks
        y_coor = self.EDGE_LENGTH*(TUBES_ALONG_Y+2)
        for rack in range(self.NUM_RACKS):
            x_coor = self.EDGE_LENGTH*((self.TUBES_ALONG_X+1)*(rack + 0.5)-0.75)
            self.draw.text((x_coor, y_coor),
                           text = str(rack+1),
                           font = rackFont,
                           fill=(0,0,0))

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
                if all([0 <= m and m < bound \
                        for m, bound in \
                        zip(row, [self.NUM_RACKS, \
                                  self.TUBES_ALONG_X, \
                                  self.TUBES_ALONG_Y])]):
                    rack, x, y = row
                    if self.tubes[rack][x][y].getStatus() != PRESENT:
                        self.tubes[rack][x][y].showAsPresent()

        #set some PRESENT tubes as TARGETs
        with open(targetsFilename, 'r') as file:
            for line in file.readlines():
                row = list(map(lambda x: int(x)-1, line.split(',')))
                #make sure all three are in bounds
                if all([0 <= m and m < bound \
                        for m, bound in \
                        zip(row, [self.NUM_RACKS, \
                                  self.TUBES_ALONG_X, \
                                  self.TUBES_ALONG_Y])]):
                    rack, x, y = row
                    #only set PRESENT tubes to TARGET
                    if self.tubes[rack][x][y].getStatus() == PRESENT:
                        self.tubes[rack][x][y].showAsTarget()
                    # else:
                    #     raise Warning("Target tube at rack {}, position {},{} not recorded as present in tray!".format(rack, x, y))


    def pickTube(self, rack, x, y):
        if x in alphabet:
            x_num = alphabet.index(x)
        elif x in lowerAlphabet:
            x_num = lowerAlphabet.index(x)
        else:
            print('Bad input')
            return

        if all([0 <= m and m < bound \
                for m, bound in zip([rack, x_num, y], \
                                    [self.NUM_RACKS, \
                                     self.TUBES_ALONG_X, \
                                     self.TUBES_ALONG_Y])]):
            target = self.tubes[rack][x_num][y]
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

    #write the tube image to output file
    def saveImage(self, filename):
        self.win.save(filename, 'PNG')

    def showImage(self):
        self.win.show()

if __name__ == '__main__':
    NUM_RACKS = 5
    EDGE_LENGTH = 25 if NUM_RACKS == 6 else 30
    TUBES_ALONG_X = 8
    TUBES_ALONG_Y = 12

    #edgeLength, numRacks, tubesAlongX, tubesAlongY
    viewer = trayStatusViewer(EDGE_LENGTH, NUM_RACKS, TUBES_ALONG_X, TUBES_ALONG_Y)

    #generate random present/absent sample input
    writeInput(300, NUM_RACKS, TUBES_ALONG_X, TUBES_ALONG_Y, 'present_input.csv')
    writeInput(50, NUM_RACKS, TUBES_ALONG_X, TUBES_ALONG_Y, 'target_input.csv')

    viewer.newTray('present_input.csv', 'target_input.csv')
    viewer.showImage()

    target = input('Enter coordinates <rack>,<x>,<y>: ')
    while target and ',' in target:
        rack, x, y = target.split(',')
        viewer.pickTube(int(rack)-1, x, TUBES_ALONG_Y - int(y))
        viewer.showImage()
        target = input('Enter coordinates <rack>,<x>,<y>: ')

    viewer.saveImage('tray.png')
