from gpiozero import OutputDevice
from time import sleep

solenoid = OutputDevice(17)

solenoid.on()
sleep(5)
solenoid.off()
