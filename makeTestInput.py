import random
EDGE_LENGTH = 30
NUM_RACKS = 5
TUBES_ALONG_X = 8
TUBES_ALONG_Y = 12
with open('testInput.csv', 'w') as file:
    for i in range(300):
        file.write('{},{},{}\n'.format(random.randint(1,NUM_RACKS), random.randint(1,TUBES_ALONG_X), random.randint(1,TUBES_ALONG_Y)))
