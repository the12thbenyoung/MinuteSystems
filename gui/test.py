from PIL import Image, ImageDraw

def imdraw(im, draw):
    draw.rectangle([(100,100),(200,200)], fill=(255,255,255), outline=(0,0,0))

im = Image.new(mode='RGB', size=(500,500), color=(255,255,255))
draw = ImageDraw.Draw(im)
imdraw(im, draw)

# write to stdout
im.save('test.png', "PNG")
