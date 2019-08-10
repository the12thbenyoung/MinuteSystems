from motor import Motor
import time

motor = Motor()

rack=0
tube=0
step=0

#Put in 8 for rack to reset. Put in 9 to stop.
while (rack != 9):
    rack = int(input("Input a rack: "))
    if (rack==8):
        motor.returnHome()
    elif (rack==7):
        motor.release()
    elif (rack==6):
        step=int(input("Input a step: "))
        if (step<0):
            motor.backward(-1*step)
        else:
            motor.forward(step)
    elif (rack<6):
        tube = int(input("Input a tube: "))
        if (tube==0):
            motor.moveToRack(rack)
        else:
            motor.moveToTube(rack, tube)
