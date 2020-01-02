from gpiozero import OutputDevice
from time import sleep

class SolenoidArray:

    #This class represents one solenoid
    class Solenoid:

        #Initializes a solenoid on a GPIO pin. Called before using a solenoid
        def __init__(self, gpio_pin):
            self.solenoid = OutputDevice(gpio_pin);

        def on(self):
            self.solenoid.on()

        def off(self):
            self.solenoid.off()

        def actuate(self):
            self.solenoid.on()
            sleep(.1)
            self.solenoid.off()
            sleep(.5)
            
        def actuateVariable(self, burst, pause):
            self.solenoid.on()
            sleep(burst)
            self.solenoid.off()
            sleep(pause)

    def __init__(self):
        self.solenoids = [
            self.Solenoid(19),
            self.Solenoid(13),
            self.Solenoid(6),
            self.Solenoid(25),
            self.Solenoid(23),
            self.Solenoid(27),
            self.Solenoid(20),
            self.Solenoid(16),
            self.Solenoid(12),
            self.Solenoid(5),
            self.Solenoid(22),
            self.Solenoid(24)
        ]

    def actuateSolenoid(self, solenoid):
        self.solenoids[solenoid].actuate()
        
    def actuateSolenoidVariable(self, solenoid, burst, pause):
        self.solenoids[solenoid].actuateVariable(burst, pause)
