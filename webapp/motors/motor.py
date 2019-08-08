from adafruit_motorkit import MotorKit

class Motor:

    BACKWARD = 2 #Constant for backwards movement from stepper class

    STEPS_TO_CENTIMETERS = 1 #This is the constant for one step to one unit of distance
    #Update this constant and test it

    def __init__(self):
        self.kit = MotorKit()
        self.stepsFromHome = 0

    #Moves both motors forward one step
    def stepForward(self):
        self.kit.stepper1.onestep()
        self.kit.stepper2.onestep()
        self.stepsFromHome += 1

    #Moves both motors backward one step
    def stepBackward(self):
        self.kit.stepper1.onestep(direction=self.BACKWARD)
        self.kit.stepper2.onestep(direction=BACKWARD)
        self.stepsFromHome -= 1

    #Moves forward (right) by a given distance
    def forward(self, centimeters):
        for i in range(centimeters*self.STEPS_TO_CENTIMETERS):
            self.stepForward()

    #Moves backward (left) by a given distance
    def backward(self, centimeters):
        for i in range(centimeters*self.STEPS_TO_CENTIMETERS):
            self.stepBackward()

    def returnHome(self):
        for i in range(self.stepsFromHome):
            self.stepBackward()

    def release(self):
        self.kit.stepper1.release()
        self.kit.stepper2.release()
