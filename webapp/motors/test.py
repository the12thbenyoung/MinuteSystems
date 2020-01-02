from motor import Motor
from time import sleep

motor = Motor()

order=""
rack=0
column=0
step=0

print("Commands: exit, zero, rel (release), step, cam (camera), col (column), pos (position)")

while (order != "exit"):
    order = input("Wus poppin jimbo? ")
    if (order == "zero"):
        motor.returnHome()
    elif (order == "rel"):
        motor.release()
    elif (order == "step"):
        step = int(input("Input a step: "))
        if (step<0):
            motor.backward(-1*step)
        else:
            motor.forward(step)
    elif (order == "cam"):
        rack = int(input("Input a rack: "))
        motor.moveToRackForCamera(rack)
    elif (order == "col"):
        rack = int(input("Input a rack: "))
        column = int(input("Input a column: "))
        motor.moveToTube(rack, column)
    elif (order == "pos"):
        print(motor.position)
    elif (order == "test"):
        motor.test()
'''
while(1):
    for i in range(1000):
        motor.stepForwardDouble()
        #sleep(.1)
    for i in range(1000):
        motor.stepBackwardDouble()
        #sleep(.1)
'''