from motor import Motor

motor = Motor()

order=""
rack=0
column=0
step=0

print("Commands: exit, zero, release, step, camera, column, position")

while (order != "exit"):
    order = input("Wus poppin jimbo? ")
    if (order == "zero"):
        motor.returnHome()
    elif (order == "release"):
        motor.release()
    elif (order == "step"):
        step = int(input("Input a step"))
        if (step<0):
            motor.backward(-1*step)
        else:
            motor.forward(step)
    elif (order == "camera"):
        rack = int(input("Input a rack: "))
        motor.moveToRackForCamera(rack)
    elif (order == "column"):
        rack = int(input("Input a rack: "))
        tube = int(input("Input a column: "))
        motor.moveToTube(rack, column)
    elif (order == "position"):
        print(motor.position)
    elif (order == "test"):
        motor.test()