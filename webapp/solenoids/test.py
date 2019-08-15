from solenoidArray import SolenoidArray

solenoidArray = SolenoidArray()

order = ""

print("Commands: exit, pop")

while (order != -1):
    order = int(input("Enter a command: "))
    #if (order == "pop"):
        #row = int(input("Enter a row: "))
    if (order != -1):
        solenoidArray.actuateSolenoid(order)