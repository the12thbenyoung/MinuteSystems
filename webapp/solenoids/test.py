from solenoidArray import SolenoidArray

solenoidArray = SolenoidArray()

order = ""

print("Commands: exit = -1, or # 0 thru 11")

while (order != -1):
    order = int(input("Enter a command: "))
    #if (order == "pop"):
        #row = int(input("Enter a row: "))
    if (order != -1):
        solenoidArray.actuateSolenoid(order)