from solenoidArray import SolenoidArray

solenoidArray = SolenoidArray()

order = ""

print("Commands: exit, pop")

while (order != "exit"):
    order = input("Enter a command: ")
    if (order == "pop"):
        solenoidArray.actuateSolenoid(0)