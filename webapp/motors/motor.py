from adafruit_motorkit import MotorKit

class Motor:

    BACKWARD = 2 #Constant for backwards movement from stepper class

    stepsFromHome = 0
    STEP_TO_DISTANCE = 1 #This is the constant for one step to one unit of distance
    #Update this constant and test it

    def __init__(self):
        self.kit = MotorKit()

    #Moves both motors forward one step
    def stepForward():
        kit.stepper1.onestep()
        kit.stepper2.onestep()
        stepsFromHome += 1

    #Moves both motors backward one step
    def stepBackward():
        kit.stepper1.onestep(direction=BACKWARD)
        kit.stepper2.onestep(direction=BACKWARD)
        stepsFromHome -= 1

    #Moves forward (right) by a given distance
    def forward(distance):
        for i in range(distance*STEPS_TO_DISTANCE):
            stepForward()

    #Moves backward (left) by a given distance
    def fbackward(distance):
        for i in range(distance*STEPS_TO_DISTANCE):
            stepBackward()
