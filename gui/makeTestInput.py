import random

def writeInput(num, numRacks, tubesAlongX, tubesAlongY, filename):
    with open(filename, 'w') as file:
        for i in range(num):
            file.write('{},{},{}\n'.format(random.randint(1,numRacks), random.randint(1,tubesAlongX), random.randint(1,tubesAlongY)))
