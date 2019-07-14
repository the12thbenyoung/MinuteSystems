from flask import Flask, flash, render_template, request, redirect, send_from_directory, url_for, send_file
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'C:/Users/noahb/Desktop/webapp/test.csv'
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
        file.save('C:/Users/noahb/Desktop/webapp/test.csv')
    return render_template('file_uploaded.html')

@app.route('/download_it')
def download_it():
    return send_file('test.csv', mimetype='text/csv',
                     attachment_filename='test.csv', as_attachment=True)

@app.route('/run_tray')
def run_tray():
    print("Runing Tray")
    return render_template('pick_tubes.html')

@app.route('/check_tray')
def check_tray():
    print("Checking Tray")
    return render_template('pick_tubes.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
