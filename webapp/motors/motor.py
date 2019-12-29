from adafruit_motorkit import MotorKit
from gpiozero import Button

class Motor:
    
    limitSwitch = Button(18)

    #Constants from stepper class
    BACKWARD = 2
    
    SINGLE = 1
    INTERLEAVED = 3
    MICROSTEP = 4

    #A step is 16 microsteps
    MICROSTEPS_PER_RACK = 6805
    MICROSTEPS_PER_TUBE = 722
    
    CAMERA_OFFSET = 1300 #TEST THIS was 2402
    RACK_OFFSET = 7045

    def __init__(self):
        self.kit = MotorKit()
        self.position = 0

    #Moves both motors forward one step
    def stepForward(self):
        self.kit.stepper1.onestep(style=self.SINGLE)
        self.kit.stepper2.onestep(style=self.SINGLE)
        #self.position += 1

    #Moves both motors backward one step
    def stepBackward(self):
        self.kit.stepper1.onestep(direction=self.BACKWARD, style=self.SINGLE)
        self.kit.stepper2.onestep(direction=self.BACKWARD, style=self.SINGLE)
        #self.position -= 1

    def stepForwardFull(self):
        self.kit.stepper1.onestep(style=self.SINGLE)
        self.kit.stepper2.onestep(style=self.SINGLE)
        
    def stepBackwardMicro(self):
        self.kit.stepper1.onestep(direction=self.BACKWARD, style=self.MICROSTEP)
        self.kit.stepper2.onestep(direction=self.BACKWARD, style=self.MICROSTEP)

    #Moves forward (right) by a given number of steps
    def forward(self, steps):
        for i in range(steps):
            self.stepForward()

    #Moves backward (left) by a given number of steps
    def backward(self, steps):
        for i in range(steps):
            self.stepBackward()

    def moveToTube(self, rack, tube):
        microsteps = (rack*self.MICROSTEPS_PER_RACK+tube*self.MICROSTEPS_PER_TUBE)-self.position+self.RACK_OFFSET
        microround = round(microsteps/16)
        if (microround<0):
            self.backward(microround*-1)
        else:
            self.forward(microround)
        self.position += microround*16

    def moveToRack(self, rack):
        microsteps = (rack*self.MICROSTEPS_PER_RACK)-self.position+self.RACK_OFFSET
        microround = round(microsteps/16)
        for i in range(microround):
            self.stepForwardFull()
        self.position += microround*16
        
    def moveToRackForCamera(self, rack):
        microsteps = (rack*self.MICROSTEPS_PER_RACK)-self.position+self.CAMERA_OFFSET
        microround = round(microsteps/16)
        if (microround<0):
            self.backward(microround*-1)
        else:
            self.forward(microround)
        self.position += microround*16

    def returnHome(self):
        steps = round(self.position/16)
        self.position -= steps*16
        for i in range(steps):
            self.stepBackward()
            if (self.limitSwitch.is_pressed == True):
                break
        while (self.limitSwitch.is_pressed == False):
            self.stepBackward()
        self.position = 0

    def release(self):
        self.kit.stepper1.release()
        self.kit.stepper2.release()
        
    def test(self):
        self.limitSwitch.wait_for_press(10)
        print("kek")
