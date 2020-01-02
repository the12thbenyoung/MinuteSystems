from motors.motor import Motor
from solenoids.solenoidArray import SolenoidArray
from time import sleep

motor = Motor()
solenoidArray = SolenoidArray()

motor.returnHome()

#for h in range(6):
for i in range(8):
    motor.moveToTube(0, i)
    for j in range(12):
        solenoidArray.actuateSolenoid(j)