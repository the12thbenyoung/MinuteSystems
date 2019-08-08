import pandas as pd
from trayStatusViewer import trayStatusViewer 

TUBES_ALONG_X = 8
TUBES_ALONG_Y = 12

inputFile = 'Example_One_Rack.csv'
inputDf = pd.read_csv(inputFile, dtype={'SampleBarcode': str})

#default to 5 racks unless there are 6 listed in file
numRacks = 6 if 6 in inputDf['RackPositionInTray'] else 5
edgeLength = 25 if numRacks == 6 else 30

#Maintains image showing the tubes in the tray which are present, 
#have already been picked, and still have to be picked
viewer = trayStatusViewer(edgeLength, numRacks, TUBES_ALONG_X, TUBES_ALONG_Y)

#columns relevant to the viewer
viewerInput = inputDf[['RackPositionInTray', 'WellID', 'Pick']]

#initialize viewer with new tray
viewer.newTray(viewerInput)
#viewer.showImage()

viewer.saveImage("../static/traydisplay.jpg")

print("kek")
