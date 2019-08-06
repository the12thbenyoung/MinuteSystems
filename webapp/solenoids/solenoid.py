from gpiozero import OutputDevice
from time import sleep

#This class represents one solenoid
class Solenoid:

    #Initializes a solenoid on a GPIO pin. Called before using a solenoid
    def __init__(self, gpio_pin):
        self.solenoid = OutputDevice(gpio_pin);

    def on():
        solenoid.on()

    def off():
        solenoid.off()

    def actuate():
        solenoid.on()
        sleep(0.5)
        solenoid.off()

