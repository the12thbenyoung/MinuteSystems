from PIL import Image, ImageDraw, ImageFont

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)
GREEN = (0,255,0)

def color_tube(draw, x, y, edge_length, color, second_color = None):
    """draw the circle representing the tube filled with given color(s)
    and black outline
    """
    draw.rectangle([(x,y), \
                         (x + edge_length, y + edge_length)], \
                        fill=WHITE, \
                        outline=(0,0,0),
                        width=2)
    if second_color:
        #split circle into 2 colors
        draw.chord([(x + edge_length/10, y + edge_length/10), \
                         (x + edge_length*9/10, y + edge_length*9/10)], \
                        90, #start at 6 (0 degrees is at 3 o'clock)
                        270, #end at noon
                        fill=color,\
                        outline=(0,0,0), \
                        width=2)
        draw.chord([(x + edge_length/10, y + edge_length/10), \
                         (x + edge_length*9/10, y + edge_length*9/10)], \
                        270, #start at noon 
                        90, #end at 6 
                        fill=second_color, \
                        outline=(0,0,0), \
                        width=2)

    else:
        #draw full cicrle with one color
        draw.ellipse([(x + edge_length/10, y + edge_length/10), \
                           (x + edge_length*9/10, y + edge_length*9/10)], \
                          fill=color, \
                          outline=(0,0,0), \
                          width=2)

def make_scan_pick_guide():
    """Make guide with all six split-tubes: absent, present, target, and the
    corresponding error states
    """
    #image that represents the whole tray
    img_x = 1300
    img_y = 112*6

    win = Image.new(mode='RGB', size=(img_x,img_y), color=(255,255,255))
    #object used to draw shapes on win
    draw = ImageDraw.Draw(win)

    draw.rectangle([(0,0),(img_x,img_y)],
                   fill=WHITE,
                   outline=BLACK,
                   width=5)

    edge_length = 100

    square_colors = [(0,0,0)]*6
    tube_colors = [WHITE,BLACK,YELLOW,(WHITE,RED),(BLACK,RED),(YELLOW,RED)]
    xs = [10]*6
    ys = [10 + 110*i for i in range(6)]

    for square_color, tube_color, x, y in zip(square_colors, tube_colors, xs, ys):
        if len(tube_color) == 2:
            color1, color2 = tube_color
            color_tube(draw, x, y, edge_length, color1, color2)
        else:
            color_tube(draw, x, y, edge_length, tube_color)


    font = ImageFont.truetype('COURIER.TTF', size=25)
    text_ys = [50 + 110*(i) for i in range(6)]
    labels = ['Confirmed absent',
              'Confirmed present',
              'Confirmed present and target to be picked',
              'Scan says absent, file says present',
              'Scan says present, file says absent or has different tube',
              'File says present and pick target, scan says absent or has different tube']
    for y, label in zip(text_ys, labels):
        draw.text((140,y),
                   text = label,
                   font = font,
                   fill = BLACK)

    win.show()
    win.save('just_scan_pick_guide.jpg', 'JPEG')

def make_scan_guide():
    """Make guide with just the 4 non-target split-tubes
    """
    #image that represents the whole tray
    img_x = 1050
    img_y = 112*4 + 5

    win = Image.new(mode='RGB', size=(img_x,img_y), color=(255,255,255))
    #object used to draw shapes on win
    draw = ImageDraw.Draw(win)

    draw.rectangle([(0,0),(img_x,img_y)],
                   fill=WHITE,
                   outline=BLACK,
                   width=5)

    edge_length = 100

    square_colors = [(0,0,0)]*6
    tube_colors = [WHITE,BLACK,(WHITE,RED),(BLACK,RED)]
    xs = [10]*4
    ys = [10 + 110*i for i in range(4)]

    for square_color, tube_color, x, y in zip(square_colors, tube_colors, xs, ys):
        if len(tube_color) == 2:
            color1, color2 = tube_color
            color_tube(draw, x, y, edge_length, color1, color2)
        else:
            color_tube(draw, x, y, edge_length, tube_color)


    font = ImageFont.truetype('COURIER.TTF', size=25)
    text_ys = [50 + 110*(i) for i in range(4)]
    labels = ['Confirmed absent',
              'Confirmed present',
              'Scan says absent, file says present',
              'Scan says present, file says absent or has different tube']
    for y, label in zip(text_ys, labels):
        draw.text((140,y),
                   text = label,
                   font = font,
                   fill = BLACK)

    win.show()
    win.save('just_scan_guide.jpg', 'JPEG')


def make_after_run_guide():
    """Guide with filled squares to show the results of picking for only target tubes
    """
    #image that represents the whole tray
    img_x = 680
    img_y = 112*3 + 5
    edge_length = 100

    win = Image.new(mode='RGB', size=(img_x,img_y), color=(255,255,255))
    #object used to draw shapes on win
    draw = ImageDraw.Draw(win)

    square_colors = [GREEN,RED,RED]
    tube_colors = [WHITE,WHITE,BLACK]
    xs = [10]*3
    ys = [10 + 110*i for i in range(3)]
    for square_color, tube_color, x, y in zip(square_colors, tube_colors, xs, ys):
        draw.rectangle([(x,y), \
                             (x + edge_length, y + edge_length)], \
                            fill=square_color, \
                            outline=(0,0,0),
                            width=2)
        draw.ellipse([(x + edge_length/10, y + edge_length/10), \
                           (x + edge_length*9/10, y + edge_length*9/10)], \
                          fill=tube_color, \
                          outline=(0,0,0),
                            width=2)

    font = ImageFont.truetype('COURIER.TTF', size=25)
    text_ys = [50 + 110*(i) for i in range(3)]
    labels = ['Tube picked successfully',
              'Tube picked, should not have been',
              'Tube not picked, should have been']
    for y, label in zip(text_ys, labels):
        draw.text((140,y),
                   text = label,
                   font = font,
                   fill = BLACK)
    win.show()
    win.save('after_run_guide.jpg', 'JPEG')

if __name__ == '__main__':
    make_scan_pick_guide()
    make_scan_guide()
    make_after_run_guide()
