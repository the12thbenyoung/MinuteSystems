from solenoidArray import SolenoidArray
from time import sleep

solenoidArray = SolenoidArray()

order = ""

print("Commands: exit = -1, or # 0 thru 11")

burst=.1
pause=.5

while (order != -1):
    order = int(input("Enter a command: "))
    #if (order == "pop"):
        #row = int(input("Enter a row: "))
    if (order <= 11 and order != -1):
        solenoidArray.actuateSolenoidVariable(order, burst, pause)
    elif (order == 12):
        burst = float(input("Enter the burst time: "))
    elif (order == 13):
        pause = float(input("Enter the pause time: "))
'''
#while(1):
for i in range(12):
    solenoidArray.actuateSolenoidVariable(i, burst, pause)
'''