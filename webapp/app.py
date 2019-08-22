from flask import Flask, flash, render_template, request, redirect, send_from_directory, url_for, send_file, session
from flask_session import Session
from werkzeug.utils import secure_filename
import time
# from solenoids.solenoidArray import SolenoidArray
# from motors.motor import Motor
import pandas as pd
from viewer.trayStatusViewer import trayStatusViewer
from numpy import unique
import os
from qr.dataMatrixDecoder import process_rack
from multiprocessing import Pool, Process, Queue

#CONSTANTS SECTION STARTS
SESSION_TYPE = 'redis'

TUBES_ALONG_X = 8
TUBES_ALONG_Y = 12

WORKING_DIRECTORY = os.getcwd()
UPLOAD_FOLDER = os.path.join(WORKING_DIRECTORY, 'viewer')
ALLOWED_EXTENSIONS = {'txt', 'csv'}

#CONSTANTS SECTION ENDS

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config.from_object(__name__)
Session(app)

# motor = Motor()

# solenoidArray = SolenoidArray()

#convert 'B08' style input from file to (x,y) index tuple - (1,7)
def convertRow(rowLetter):
    alphabet = 'ABCDEFGH'
    return alphabet.index(rowLetter)

#This method scans a tray
#It needs to be told how many rack there are in a tray
def scan(num_racks):
    """scans the current tray containing num_rack racks.
    returns a queue of results, each element containing
    (rack number, filename, data: {hash((col,row)): tube number}, number of tubes located, number of tubes located)
    """
    process_list = []
    data_queue = Queue()
    
    for i in range(num_racks):
        # motor.moveToRackForCamera(i)

        imagePath = 'qr/shpongus.jpg'
        # imagePath = os.path.join(WORKING_DIRECTORY, f'camerapics/rack{i}.jpg')
        # camera.start_preview()
        # sleep(2)
        # camera.capture(imagePath)
        # camera.stop_preview()

        proc = Process(target=process_rack, args=(i,imagePath,data_queue))
        proc.start()
        process_list.append(proc)
        
    # motor.returnHome()
    # motor.release()
    
    for proc in process_list:
        proc.join()
    
    #found_data_list = []
    #while not data_queue.empty():
    #    filename, dataIndices, tubeFound, matricesDecoded = data_queue.get()
    #    found_data_list.append([filename, tubesFound, matricesDecoded])
    
    #To grab data at row,col do data_indices[hash((col,row))]
    
    return data_queue

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def nextTray(viewer, tray_id, tray_data):
    #columns relevant to the viewer
    viewerInput = tray_data[['RackPositionInTray', 'WellID', 'Pick']]

    #initialize viewer with new tray
    viewer.new_tray(viewerInput)
    
@app.route('/')
def index():
    #truncate output csv file and write headers
    with open(os.path.join(UPLOAD_FOLDER, 'output.csv'), 'w') as f:
        f.write('TrayID,RackID,RackPositionInTray,WellID,SampleBarcode\n')

    return render_template('index.html')

@app.route('/picking_begin')
def picking_begin():
    return render_template('picking/upload_file.html')

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

        tray_ids = list(unique(inputDataframe['TrayID']))
        #data frames holding the rows associated with each tray
        tray_dataframes = [inputDataframe[inputDataframe['TrayID'] == tray_id] for tray_id in tray_ids]
        #holds number of racks in each tray (5 or 6)
        num_racks_list = [(6 if 5 in tray_df['RackPositionInTray'] else 5) for tray_df in tray_dataframes]

        tray_data_zip = list(zip(tray_ids, tray_dataframes, num_racks_list))
        session['tray_data_list'] = tray_data_zip

        for i, (tray_id, tray_dataframe, num_racks) in enumerate(tray_data_zip):
            #Maintains image showing the tubes in the tray which are present, 
            #have already been picked, and still have to be picked
            edge_length = 25 if num_racks == 6 else 30
            viewer = trayStatusViewer(edge_length, num_racks, TUBES_ALONG_X, TUBES_ALONG_Y)
            viewer.new_tray(tray_dataframe[tray_dataframe['TrayID'] == tray_id])
            #save image of tray in 'static/images/' to to be shown in file_uploaded.html
            viewer.save_image(os.path.join(WORKING_DIRECTORY, f'static/images/traydisplay{i}.jpg'))
            if i == 0:
                #only show first image in pick_tubes
                viewer.save_image(os.path.join(WORKING_DIRECTORY, 'static/traydisplay.jpg'))
                # num_racks = self.globalNumRacks
                
        #current_tray_num is the index of tray_data_list with the current tray's data
        session['current_tray_num'] = 0

        if 'tray_data_list' in session \
        and len(session['tray_data_list']) > 0 \
        and len(session['tray_data_list'][0]) > 0:
            next_tray_id = session['tray_data_list'][0][0]
        else:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            next_tray_id = None

    return render_template('picking/scan_tray.html', nextTrayId = next_tray_id)

#This is run after the user presses Scan Tray on scan_tray.html
#Or after the user presses Scan Again on tray_scanned.html
#This is the first scan. It is done before the user runs the tray
@app.route('/picking_scan_tray')
def picking_scan_tray():
    tray_id, file_data, num_racks = session['tray_data_list'][session['current_tray_num']]
    edge_length = 25 if num_racks == 6 else 30

    viewer = trayStatusViewer(edge_length, num_racks, TUBES_ALONG_X, TUBES_ALONG_Y)
    viewer.new_tray(file_data)

    #The tray is scanned and the info is returned into data_queue
    scan_data_queue = scan(num_racks)

    #You need to make the tray image (save it to the normal spot, static/traydisplay.jpg)
    #and set desired_tubes_incorrect and total_tubes_correct
    #this might be run multiple times so it needs to be able to handle that
    img_save_path = os.path.join(WORKING_DIRECTORY, 'static/traydisplay.jpg')

    scan_data_by_rack, desired_tubes_incorrect, total_tubes_incorrect = viewer.make_pre_run_scan_results(scan_data_queue, \
                                                                                      file_data, \
                                                                                      num_racks, \
                                                                                      img_save_path)
    
    #save returned scan data dict to memory so we can tell what changed after running tray
    session['prev_scan_data'] = scan_data_by_rack

    return render_template('picking/tray_scanned.html',
                           desiredTubesIncorrect = desired_tubes_incorrect, totalTubesIncorrect = total_tubes_incorrect)

@app.route('/abort_order', methods=['GET', 'POST'])
def download():
    return send_file(os.path.join(UPLOAD_FOLDER, 'output.csv'), mimetype='text/csv',
                     attachment_filename='output.csv', as_attachment=True)

@app.route('/skip_tray')
def skip_tray():
    #Skip it
    return render_template('picking/scan_tray.html')

#This is run after the user presses Run Tray on tray_scanned.html'
#This does the second scan which is done after the tray is ran
@app.route('/run_tray')
def run_tray():
    tray_data_list = session.get('tray_data_list', None)
    if tray_data_list:
        #run the current tray
        tray_id, tray_data, num_racks = tray_data_list[session['current_tray_num']]
        
        edge_length = 25 if num_racks == 6 else 30
        viewer = trayStatusViewer(edge_length, num_racks, TUBES_ALONG_X, TUBES_ALONG_Y)
        viewer.new_tray(tray_data)

        tray_data_pick = tray_data[tray_data['Pick'] == 1]
        racks = unique(tray_data_pick['RackPositionInTray'])
        racks.sort()
        for rack_index in racks:
            rackData = tray_data_pick[tray_data_pick['RackPositionInTray'] == rack_index]
            columns = unique(rackData['TubeColumn'])
            columns.sort()
            for col in columns:
                #move to rack,column
                # motor.moveToTube(int(rack_index), int(col))
                colData = rackData[rackData['TubeColumn'] == col]
                for row in colData['TubeRow']:
                    print(tray_id, rack_index, col, row)
                    #activate soleniod
                    # solenoidArray.actuateSolenoid(int(row))

        # motor.returnHome()
        # motor.release()
        
        scan_data_queue = scan(num_racks)
        
        #list of rack ids - index is rack location in tray
        rack_ids = []
        for rack_index in range(num_racks):
            rack_ids.append(tray_data[tray_data['RackPositionInTray'] == rack_index].iloc[0]['RackID'])

        #You need to make the tray image (save it to the normal spot, static/traydisplay.jpg)
        #and set running_errors somewhere in here
        running_errors = viewer.make_post_run_scan_results(scan_data_queue, \
                                                           tray_data_pick, \
                                                           session['prev_scan_data'], \
                                                           num_racks, \
                                                           tray_id,
                                                           rack_ids,
                                                           os.path.join(WORKING_DIRECTORY, 'static/traydisplay.jpg'), \
                                                           os.path.join(UPLOAD_FOLDER, 'output.csv'))

    return render_template('picking/tray_ran.html', runningErrors = running_errors)

#This is run after the user presses Scan Again on tray_ran.html
@app.route('/picking_rescan_tray')
def picking_rescan_tray():
    tray_id, file_data, num_racks = session['tray_data_list'][session['current_tray_num']]
    edge_length = 25 if num_racks == 6 else 30

    viewer = trayStatusViewer(edge_length, num_racks, TUBES_ALONG_X, TUBES_ALONG_Y)
    viewer.new_tray(file_data)

    scan_data_queue = scan(num_racks) #gotta grab num_racks here somehow
    
    #You need to make the tray image (save it to the normal spot, static/traydisplay.jpg)
    #and set running_errors somewhere in here
    running_errors = 0
    
    return render_template('picking/tray_ran.html', runningErrors = running_errors)

@app.route('/next_tray')
def next_tray():
    #increment to next tray
    session['current_tray_num'] += 1

    print("Next Tray")
    #if we're out of tray_data, tell user tray is done
    if session['current_tray_num'] >= len(session['tray_data_list']):
        return render_template('picking/order_complete.html')

    tray_data_list = session['tray_data_list']
    tray_id, tray_data, num_racks = tray_data_list[session['current_tray_num']]

    edge_length = 25 if num_racks == 6 else 30

    viewer = trayStatusViewer(edge_length, num_racks, TUBES_ALONG_X, TUBES_ALONG_Y)
    viewer.new_tray(tray_data)

    #save image of tray in 'static/' to to be shown in run_tray
    viewer.save_image(os.path.join(WORKING_DIRECTORY, 'static/traydisplay.jpg'))

    return render_template('picking/scan_tray.html', nextTrayId = tray_id)

@app.route('/scanning_begin')
def scanning_being():
    #Begin it
    return render_template('scanning/enter_ids.html')

@app.route('/scanning_enter_ids', methods=['POST'])
def scanning_enter_ids():
    #Enter it
    tray_id = request.form['trayid']
    rack_id0 = request.form['rackid0']
    rack_id1 = request.form['rackid1']
    rack_id2 = request.form['rackid2']
    rack_id3 = request.form['rackid3']
    rack_id4 = request.form['rackid4']
    rack_id5 = request.form['rackid5']
    
    return render_template('scanning/check_ids.html', trayid = tray_id, rackId0 = rack_id0, rackId1 = rack_id1, \
                           rackId2 = rack_id2, rackId3 = rack_id3, rackId4 = rack_id4, rackId5 = rack_id5)

#This is run after the user presses Scan Tray on check_ids.html
#Or after the user presses Rescan Tray on tray_scanned.html
@app.route('/scan_tray<int:trayId>/<int:rackId0>/<int:rackId1>/<int:rackId2>/<int:rackId3>/<int:rackId4>/<int:rackId5>',
           methods=['GET','POST'])
def scan_tray(trayId=None,rackId0=None,rackId1=None,rackId2=None,rackId3=None,rackId4=None,rackId5=None):
    num_racks = 6 if rackId5 else 5
    edge_length = 25 if num_racks == 6 else 30

    rack_ids = [id for id in [rackId0,rackId1,rackId2,rackId3,rackId4,rackId5] if id]

    scan_data_queue = scan(num_racks)

    viewer = trayStatusViewer(edge_length, num_racks, TUBES_ALONG_X, TUBES_ALONG_Y)
    viewer.new_tray()
    #makes image and writes to output csv file
    viewer.make_just_scan_results(scan_data_queue, \
                                  trayId, \
                                  rack_ids, \
                                  os.path.join(WORKING_DIRECTORY, 'static/traydisplay.jpg'),\
                                  os.path.join(UPLOAD_FOLDER, 'output.csv'))\

    return render_template('scanning/scanning_tray_scanned.html')

@app.route('/scanning_download_csv')
def scanning_download_csv():
    return send_file(os.path.join(UPLOAD_FOLDER, 'output.csv'), mimetype='text/csv',
                     attachment_filename='output.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)#, host='0.0.0.0')
