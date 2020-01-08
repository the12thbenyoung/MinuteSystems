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
        '''
        def actuate(self):
            self.solenoid.on()
            sleep(self.burst)
            self.solenoid.off()
            sleep(delay)
        '''
        def actuateVariable(self, burst, delay):
            self.solenoid.on()
            sleep(burst)
            self.solenoid.off()
            sleep(delay)

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
        f = open("/home/pi/Documents/MinuteSystems/webapp/solenoids/config.txt", "r")
        line = f.readline()
        self.burst = float(line[-5:-1])
        print(self.burst)
        line = f.readline()
        self.delay = float(line[-5:-1])
        print(self.delay)
        f.close()

    def actuateSolenoid(self, solenoid):
        self.solenoids[solenoid].actuateVariable(self.burst, self.delay)
        
    def actuateSolenoidVariable(self, solenoid, burst, delay):
        self.solenoids[solenoid].actuateVariable(burst, delay)
