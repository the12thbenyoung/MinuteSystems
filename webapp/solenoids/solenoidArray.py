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
            sleep(5)
            self.solenoid.off()

    def __init__(self):
        self.solenoids = [
            self.Solenoid(17),
            self.Solenoid(18),
            self.Solenoid(27),
            self.Solenoid(22),
            self.Solenoid(23),
            self.Solenoid(24),
            self.Solenoid(5),
            self.Solenoid(6),
            self.Solenoid(12),
            self.Solenoid(13),
            self.Solenoid(16),
            self.Solenoid(19)
        ]

    def actuateSolenoid(self, solenoid):
        self.solenoids[solenoid].actuate()
