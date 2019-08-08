from flask import Flask, flash, render_template, request, redirect, send_from_directory, url_for, send_file
from werkzeug.utils import secure_filename
import time
from adafruit_motorkit import MotorKit
from solenoids.solenoid import Solenoid
from motors.motor import Motor
import pandas as pd
from prod.trayStatusViewer import trayStatusViewer

solenoid4 = Solenoid(17)

#motor = Motor()


TUBES_ALONG_X = 8
TUBES_ALONG_Y = 12

inputFile = '/home/pi/MinuteSystems/webapp/test.csv'
inputDf = pd.read_csv(inputFile, dtype={'SampleBarcode': str})
#default to 5 racks unless there are 6 listed in file
numRacks = 6 if 6 in inputDf['RackPositionInTray'] else 5
edgeLength = 25 if numRacks == 6 else 30

#Maintains image showing the tubes in the tray which are present, 
#have already been picked, and still have to be picked
viewer = trayStatusViewer(edgeLength, numRacks, TUBES_ALONG_X, TUBES_ALONG_Y)


UPLOAD_FOLDER = '/home/pi/MinuteSystems/webapp/test.csv'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pick_tubes')
def pick_tubes():
    return render_template('pick_tubes.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/get_csv_file', methods=['GET', 'POST'])
def get_csv_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save('/home/pi/MinuteSystems/webapp/test.csv')

        #columns relevant to the viewer
        viewerInput = inputDf[['RackPositionInTray', 'WellID', 'Pick']]

        #initialize viewer with new tray
        viewer.newTray(viewerInput)
        #viewer.showImage()

        viewer.saveImage("/home/pi/MinuteSystems/webapp/static/kek.jpg")
    return render_template('file_uploaded.html')

@app.route('/download_it')
def download_it():
    return send_file('test.csv', mimetype='text/csv',
                     attachment_filename='test.csv', as_attachment=True)

@app.route('/run_tray')
def run_tray():
    viewer.pickTube(1, 'H', 9)
        
    return render_template('pick_tubes.html')

@app.route('/check_tray')
def check_tray():
    print("Checking Tray")
    return render_template('pick_tubes.html')

if __name__ == '__main__':
    app.run(debug=True)#, host='0.0.0.0')
