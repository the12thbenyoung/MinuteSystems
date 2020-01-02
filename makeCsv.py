import pyautogui as pag

pag.FAILSAFE = True
pag.PAUSE = 0.1

Rack = 0
WellLetter = 'A'
WellZero = 0
WellNumber = 0
WellID = WellLetter + str(WellZero) + str(WellNumber)
Pick = 1
line = f"1,1,01-01,1111111111,{Rack},{Rack},{WellID},1,1111111111,{Pick}\n"

alphabet = ['A','B','C','D','E','F','G','H']

pag.click(200,200)

print(line)

for i in range(6):
    Rack = i + 1
    if i == 1:
        Pick = 0
    elif i == 5:
        Pick = 1
    for k in range(8):
        WellLetter = alphabet[k]
        for j in range(12):
            WellNumber = j + 1
            if (WellNumber < 10):
                WellID = WellLetter + str(WellZero) + str(WellNumber)
            else:
                WellID = WellLetter + str(WellNumber)
            line = f"1,1,01-01,1111111111,{Rack},{Rack},{WellID},1,1111111111,{Pick}\n"
            pag.write(line)