import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

#tube statuses
ABSENT = 0
ABSENT_WRONG = 10
PRESENT = 1
PRESENT_WRONG = 11
TARGET = 2
TARGET_WRONG = 12
PICKED = 3
PICKED_WRONG = 13
NOT_PICKED = 4

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)
GREEN = (0,255,0)

#font to use for index labels
indexFont = ImageFont.truetype('COURIER.TTF', size=17)
#bigger font to use for rack labels
rackFont = ImageFont.truetype('COURIER.TTF', size=25)

alphabet = 'ABCDEFGH'
lowerAlphabet = 'abcdefgh'

class trayStatusViewer:
    #represents one tube - one square/circle displayed in the window
    class Tube:

        def __init__(self, x, y, edge_length, draw):
            self.x = x
            self.y = y
            self.status = ABSENT
            self.draw = draw
            self.edge_length = edge_length

            # draw empty rectangle and circle to represent tube
            draw.rectangle([(x, y), \
                            (x + edge_length, y + edge_length)], \
                           fill=(255,255,255), \
                           outline=BLACK, \
                           width=1)
            draw.ellipse([(x + edge_length/10, y + edge_length/10), \
                          (x + edge_length*9/10, y + edge_length*9/10)], \
                         fill=(255,255,255), \
                         outline=BLACK, \
                         width=1)

        def __color_tube(self, color, second_color = None):
            """draw the circle representing the tube filled with given color(s)
            and black outline
            """
            if second_color:
                #split circle into 2 colors
                self.draw.chord([(self.x + self.edge_length/10, self.y + self.edge_length/10), \
                                 (self.x + self.edge_length*9/10, self.y + self.edge_length*9/10)], \
                                90, #start at 6 (0 degrees is at 3 o'clock)
                                270, #end at noon
                                fill=color,\
                                outline=BLACK, \
                                width=1)
                self.draw.chord([(self.x + self.edge_length/10, self.y + self.edge_length/10), \
                                 (self.x + self.edge_length*9/10, self.y + self.edge_length*9/10)], \
                                270, #start at noon 
                                90, #end at 6 
                                fill=second_color, \
                                outline=BLACK, \
                                width=1)

            else:
                #draw full cicrle with one color
                self.draw.ellipse([(self.x + self.edge_length/10, self.y + self.edge_length/10), \
                                   (self.x + self.edge_length*9/10, self.y + self.edge_length*9/10)], \
                                  fill=color, \
                                  outline=BLACK, \
                                  width=1)

        def __color_tube_and_square(self, tube_color, square_color):
            """Color background square one color and tube another. Used to show correctly/
            incorrectly picked tubes
            """
            self.draw.rectangle([(self.x, self.y), \
                                 (self.x + self.edge_length, self.y + self.edge_length)], \
                                fill=square_color, \
                                outline=BLACK, \
                                width=1)
            self.draw.ellipse([(self.x + self.edge_length/10, self.y + self.edge_length/10), \
                               (self.x + self.edge_length*9/10, self.y + self.edge_length*9/10)], \
                              fill=tube_color, \
                              outline=BLACK, \
                              width=1)


        def show_as_absent(self):
            self.__color_tube(WHITE)
            self.status = ABSENT

        def show_as_absent_wrong(self):
            """file says tube should be absent, but it's present
            """
            self.__color_tube(WHITE,RED)
            self.status = ABSENT_WRONG

        def show_as_present(self):
            self.__color_tube(BLACK)
            self.status = PRESENT

        def show_as_present_wrong(self):
            """file says tube should be present, but it's absent
            """
            self.__color_tube(BLACK,RED)
            self.status = PRESENT_WRONG

        def show_as_target(self):
            self.__color_tube(YELLOW)
            self.status = TARGET

        def show_as_target_wrong(self):
            """file says tube should be target (and thus also present), but it's absent
            """
            #show as half yellow, half red
            self.__color_tube(YELLOW,RED)
            self.status = TARGET_WRONG

        def show_as_picked(self):
            self.__color_tube_and_square(WHITE,GREEN)
            self.status = PICKED

        def show_as_picked_wrong(self):
            """Tube was picked, but should not have been
            """
            #white with red outline
            self.__color_tube_and_square(WHITE,RED)
            self.status = PICKED_WRONG

        def show_as_not_picked(self):
            """Tube was not picked, but should have been
            """
            #black with red outline
            self.__color_tube_and_square(BLACK,RED)
            self.status = NOT_PICKED

        def get_status(self):
            return self.status

    def __init__(self, edge_length, num_racks, tubes_along_x, tubes_along_y):
        self.EDGE_LENGTH = edge_length
        self.NUM_RACKS = num_racks
        self.TUBES_ALONG_X = tubes_along_x
        self.TUBES_ALONG_Y = tubes_along_y

        windowX = self.EDGE_LENGTH*(self.NUM_RACKS*(self.TUBES_ALONG_X+1))
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
                    x_coor = self.EDGE_LENGTH*(rack*(self.TUBES_ALONG_X+1) + x + 1)
                    #lower indices are on the bottom (higher y coor), so start at 
                    #bottom and subtract to move up as y index increases
                    y_coor = self.EDGE_LENGTH*(self.TUBES_ALONG_Y-y)
                    y_list.append(self.Tube(x_coor, y_coor, self.EDGE_LENGTH, self.draw))
                x_list.append(y_list)
            tubes.append(x_list)
        self.tubes = tubes

        #draw column indices between each pair of trays
        for rack in range(self.NUM_RACKS):
            x_coor = int(np.round(self.EDGE_LENGTH*(rack*(self.TUBES_ALONG_X+1)+0.2)))
            single_digit_x_coor = x_coor + int(np.round(self.EDGE_LENGTH*0.18))
            for row in range(self.TUBES_ALONG_Y):
                #if row is a singe digit, shift the text a bit to the right
                mod_x_coor = single_digit_x_coor if self.TUBES_ALONG_Y-row < 10 else x_coor
                y_coor = int(np.round(self.EDGE_LENGTH*(row + 1.35)))
                self.draw.text((mod_x_coor,y_coor),
                               text = str(self.TUBES_ALONG_Y-row),
                               font = indexFont,
                               fill = (0,0,0))

        #draw row indices above each rack
        bottom_y_coor = self.EDGE_LENGTH*(self.TUBES_ALONG_Y+1.2)
        for rack in range(self.NUM_RACKS):
            for col in range(self.TUBES_ALONG_X):
                x_coor = self.EDGE_LENGTH*(rack*(self.TUBES_ALONG_X+1) + col + 1.42)
                self.draw.text((x_coor, bottom_y_coor),
                               text = alphabet[col],
                               font = indexFont,
                               fill=(0,0,0))

        #draw rack numbers beneath racks
        y_coor = self.EDGE_LENGTH*(self.TUBES_ALONG_Y+2)
        for rack in range(self.NUM_RACKS):
            x_coor = self.EDGE_LENGTH*((self.TUBES_ALONG_X+1)*(rack + 0.5)+0.25)
            self.draw.text((x_coor, y_coor),
                           text = str(rack+1),
                           font = rackFont,
                           fill=(0,0,0))

    #determine which tubes start as ABSENT, PRESENT, or TARGET and color accordingly
    def new_tray(self, locationData):
        #reset all tubes to ABSENT
        for rack in self.tubes:
            for col in rack:
                for tube in col:
                    tube.show_as_absent()

        for _, row in locationData.iterrows():
            (x,y) = int(row['TubeColumn']), int(row['TubeRow'])
            rack = int(row['RackPositionInTray'])
            toPick = row['Pick'] 
            #make sure all are in bounds
            #and pick status isn't NaN - if it is tube should stay absent
            if all([0 <= val and val < bound \
                    for val, bound in \
                    zip([rack, x, y],\
                        [self.NUM_RACKS, self.TUBES_ALONG_X, self.TUBES_ALONG_Y])]) \
                and pd.notnull(toPick):

                if int(toPick) == 1:
                    self.tubes[rack][x][y].show_as_target()
                else:
                    self.tubes[rack][x][y].show_as_present()

    def pickTube(self, rack, x, y):
        if all([0 <= m and m < bound \
                for m, bound in zip([rack, x, y], \
                                    [self.NUM_RACKS, \
                                     self.TUBES_ALONG_X, \
                                     self.TUBES_ALONG_Y])]):
            target = self.tubes[rack][x][y]
            if target.get_status() == TARGET:
                target.show_as_picked()
            elif target.get_status() == ABSENT:
                print("Tube is not in rack!")
            elif target.get_status() == PRESENT:
                print("Tube was not supposed to be picked!")
            elif target.get_status() == PICKED:
                print("Tube has already been picked!")
        else:
            print("Tube out of bounds!")

    def get_tube(self, rack, x, y):
        """get reference to tube in rack rack at index x,y
        """
        return self.tubes[rack][x][y]

    def save_image(self, filename):
        """write the tube image to output file
        """
        self.win.save(filename, 'JPEG')

    def show_tray(self):
        self.win.show()

if __name__ == '__main__':
    NUM_RACKS = 5
    EDGE_LENGTH = 25 if NUM_RACKS == 6 else 30
    TUBES_ALONG_X = 8
    TUBES_ALONG_Y = 12

    #edge_length, num_racks, tubes_along_x, tubes_along_y
    viewer = trayStatusViewer(EDGE_LENGTH, NUM_RACKS, TUBES_ALONG_X, TUBES_ALONG_Y)

    ##generate random present/absent sample input
    #writeInput(300, NUM_RACKS, TUBES_ALONG_X, TUBES_ALONG_Y, 'present_input.csv')
    #writeInput(50, NUM_RACKS, TUBES_ALONG_X, TUBES_ALONG_Y, 'target_input.csv')
    test_input = pd.DataFrame({'TubeColumn': [3,1,2,4,4], \
                               'TubeRow': [7,4,2,6,10], \
                               'RackPositionInTray': [2,3,0,1,4],
                               'Pick': [0,0,1,1,0]})

    viewer.new_tray(test_input)
    # viewer.show_tray()
    viewer.get_tube(0,0,0).show_as_absent()
    viewer.get_tube(0,0,1).show_as_absent_wrong()
    viewer.get_tube(0,0,2).show_as_present()
    viewer.get_tube(0,0,3).show_as_present_wrong()
    viewer.get_tube(0,0,4).show_as_target()
    viewer.get_tube(0,0,5).show_as_target_wrong()
    viewer.get_tube(0,0,6).show_as_picked()
    viewer.get_tube(0,0,7).show_as_picked_wrong()
    viewer.get_tube(0,0,8).show_as_not_picked()

    viewer.show_tray()
    # target = input('Enter coordinates <rack>,<x>,<y>: ')
    # while target and ',' in target:
    #     rack, x, y = target.split(',')
    #     viewer.pickTube(int(rack)-1, x, TUBES_ALONG_Y - int(y))
    #     viewer.show_tray()
    #     target = input('Enter coordinates <rack>,<x>,<y>: ')

    # viewer.save_image('tray.png')
