from flask import Flask, flash, render_template, request, redirect, send_from_directory, url_for, send_file, session
from flask_session import Session
from werkzeug.utils import secure_filename
import time
from solenoids.solenoidArray import SolenoidArray
from motors.motor import Motor
import pandas as pd
from prod.trayStatusViewer import trayStatusViewer
from numpy import unique
import os


#CONSTANTS SECTION STARTS
SESSION_TYPE = 'redis'

TUBES_ALONG_X = 8
TUBES_ALONG_Y = 12

WORKING_DIRECTORY = os.getcwd()
UPLOAD_FOLDER = os.path.join(WORKING_DIRECTORY, 'prod')
ALLOWED_EXTENSIONS = {'txt', 'csv'}

#CONSTANTS SECTION ENDS

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config.from_object(__name__)
Session(app)

motor = Motor()

solenoidArray = SolenoidArray()

#convert 'B08' style input from file to (x,y) index tuple - (1,7)
def convertRow(rowLetter):
    alphabet = 'ABCDEFGH'
    return alphabet.index(rowLetter)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def nextTray(viewer, trayId, trayData):
    #columns relevant to the viewer
    viewerInput = trayData[['RackPositionInTray', 'WellID', 'Pick']]

    #initialize viewer with new tray
    viewer.newTray(viewerInput)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pick_tubes')
def pick_tubes():
    if 'trayDataList' in session \
        and len(session['trayDataList']) > 0 \
        and len(session['trayDataList'][0]) > 0:
        next_tray_id = session['trayDataList'][0][0]
    else:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        next_tray_id = None

    return render_template('pick_tubes.html', nextTrayId = next_tray_id)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/get_csv_file', methods=['GET', 'POST'])
def get_csv_file():
    os.system('rm ' + os.path.join(WORKING_DIRECTORY, 'static/traydisplay.jpg'))
    os.system('rm ' + os.path.join(WORKING_DIRECTORY, 'static/images/*.jpg'))

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, 'user_upload.csv'))

        #Bens stuff
        inputFile = os.path.join(UPLOAD_FOLDER, 'user_upload.csv')
        inputDataframe = pd.read_csv(inputFile, dtype={'SampleBarcode': str})
        #remove rows where WellID is NaN
        inputDataframe.dropna(subset=['WellID'], axis='rows', inplace=True)
        #subtract 1 from each RackPositionInTray
        inputDataframe['RackPositionInTray'] = inputDataframe['RackPositionInTray'] \
                                                .apply(lambda x: x - 1)
        #split tubePositionsList into columns (first letter) and rows (second two numbers)
        tubePositionsList = [[convertRow(row[0]),int(row[1:])-1] for row in inputDataframe['WellID']]
        tubePositionsDf = pd.DataFrame(tubePositionsList, columns = ['TubeColumn', 'TubeRow'])

        #remove old position column from original dataframe and append new columns
        inputDataframe = inputDataframe.drop(columns=['WellID'])
        inputDataframe = pd.concat([inputDataframe, tubePositionsDf], axis=1)

        trayIds = list(unique(inputDataframe['TrayID']))
        #data frames holding the rows associated with each tray
        trayDataframes = [inputDataframe[inputDataframe['TrayID'] == trayId] for trayId in trayIds]
        #holds number of racks in each tray (5 or 6)
        numRacksList = [(6 if 5 in tray_df['RackPositionInTray'] else 5) for tray_df in trayDataframes]

        trayDataZip = list(zip(trayIds, trayDataframes, numRacksList))
        session['trayDataList'] = trayDataZip

        for i, (trayId, trayDataframe, numRacks) in enumerate(trayDataZip):
            #Maintains image showing the tubes in the tray which are present, 
            #have already been picked, and still have to be picked
            edgeLength = 25 if numRacks == 6 else 30
            viewer = trayStatusViewer(edgeLength, numRacks, TUBES_ALONG_X, TUBES_ALONG_Y)
            viewer.newTray(trayDataframe[trayDataframe['TrayID'] == trayId])
            #save image of tray in 'static/images/' to to be shown in file_uploaded.html
            viewer.saveImage(os.path.join(WORKING_DIRECTORY, f'static/images/traydisplay{i}.jpg'))
            if i == 0:
                #only show first image in pick_tubes
                viewer.saveImage(os.path.join(WORKING_DIRECTORY, 'static/traydisplay.jpg'))

    return render_template('file_uploaded.html')

@app.route('/download_it')
def download_it():
    return send_file('output.csv', mimetype='text/csv',
                     attachment_filename='output.csv', as_attachment=True)

@app.route('/run_tray')
def run_tray():
    trayDataList = session.get('trayDataList', None)
    if trayDataList:
        #run the first tray and remove it
        tray, trayData, numRacks = trayDataList.pop(0)

        edgeLength = 25 if numRacks == 6 else 30
        viewer = trayStatusViewer(edgeLength, numRacks, TUBES_ALONG_X, TUBES_ALONG_Y)
        viewer.newTray(trayData)

        trayDataPick = trayData[trayData['Pick'] == 1]
        racks = unique(trayDataPick['RackPositionInTray'])
        racks.sort()
        for rackId in racks:
            rackData = trayDataPick[trayDataPick['RackPositionInTray'] == rackId]
            columns = unique(rackData['TubeColumn'])
            columns.sort()
            for col in columns:
                #move to rack,column
                motor.moveToTube(int(rackId), int(col))
                colData = rackData[rackData['TubeColumn'] == col]
                for row in colData['TubeRow']:
                    print(tray, rackId, col, row)
                    #activate soleniod
                    solenoidArray.actuateSolenoid(int(row))
                    viewer.pickTube(rackId, col, row)

        #save image of tray in 'static/images/' to to be shown in file_uploaded.html
        viewer.saveImage(os.path.join(WORKING_DIRECTORY, 'static/traydisplay.jpg'))

        motor.returnHome()
                
    return render_template('next_tray.html')

@app.route('/check_tray')
def check_tray():
    print("Checking Tray")
    return render_template('pick_tubes.html')

@app.route('/next_tray')
def next_tray():
    print("Next Tray")
    trayDataList = session['trayDataList']
    #if we're out of trayData, tell user tray is done
    if len(trayDataList) == 0:
        return render_template('done_with_file.html')

    trayId, trayData, numRacks = trayDataList[0]

    edgeLength = 25 if numRacks == 6 else 30
    viewer = trayStatusViewer(edgeLength, numRacks, TUBES_ALONG_X, TUBES_ALONG_Y)
    viewer.newTray(trayData)

    #save image of tray in 'static/' to to be shown in run_tray
    viewer.saveImage(os.path.join(WORKING_DIRECTORY, 'static/traydisplay.jpg'))

    return render_template('pick_tubes.html', nextTrayId = trayId)

if __name__ == '__main__':
    app.run(debug=True)#, host='0.0.0.0')
