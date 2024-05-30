from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from os.path import join, dirname, realpath
from noise_add import noise_addd
from model_test import model_check

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

accuracy = ("","")
# Root URL
@app.route('/')
def index():
     # Set The upload HTML template '\templates\home.html'
    print("hi, ", accuracy) 
    return render_template('page2.html', accuracy1 = accuracy[0], accuracy2 = accuracy[1])

csvfile = ""
# Get the uploaded files
def uploadFiles(request):
    global csvfile
      # get the uploaded file
    uploaded_file = request.files['csvfile']
    csvfile = uploaded_file.filename
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # set the file path
        uploaded_file.save(file_path)
        # save the file
    
def uploadData(request):
    feature1 = {}

    col_private_info = request.form['private']
    col_binary_vals = request.form['binary']
    categorical = request.form['categorical']
    numerical = request.form['numerical']
    epsilon = request.form['epsilon']

    col_private_info = [x.strip() for x in col_private_info.split(',')]
    col_binary_vals = [x.strip() for x in col_binary_vals.split(',')]
    categorical = [x.strip() for x in categorical.split(',')]
    numerical = [x.strip() for x in numerical.split(',')]
    epsilon = [x.strip() for x in epsilon.split(',')]

    for i in range(len(epsilon)):
        epsilon[i] = float(epsilon[i])
    
    feature1['private'] = col_private_info
    feature1['binary'] = col_binary_vals
    feature1['categorical'] = categorical
    feature1['numerical'] = numerical
    feature1['epsilon'] = epsilon
    
    noise_addd(feature1, csvfile)
    path = f"files/private_{csvfile}"
    print(path)
    return send_file(path, as_attachment=True)


def uploadModel(request):
    global accuracy
    feature2 = {}
    colinp = request.form['colinp']
    colop = request.form['colop']
    mlalgo = request.form['mlalgo']
    traintest = request.form['traintest']
    mlpara = request.form['mlpara']

    colinp = [x.strip() for x in colinp.split(',')]
    colop = [x.strip() for x in colop.split(',')]
    mlalgo = [x.strip() for x in mlalgo.split(',')]
    traintest = [x.strip() for x in traintest.split(',')]
    mlpara = [x.strip() for x in mlpara.split(',')]

    mlalgo = int(mlalgo[0])

    for i in range(len(traintest)):
        traintest[i] = int(traintest[i])

    for i in range(len(mlpara)):
        mlpara[i] = float(mlpara[i])
    
    feature2['colinp'] = colinp
    feature2['colop'] = colop
    feature2['mlalgo'] = mlalgo
    feature2['traintest'] = traintest
    feature2['mlpara'] = mlpara
    accuracy = model_check(feature2, csvfile)
    

@app.route("/", methods=['POST'])
def mainapp():
    uploadFiles(request)
    try:
        a = request.form['private']
        file = uploadData(request)
        return file
    except:
        uploadModel(request)
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download(filename):
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = f"files/{filename}.csv"
    return send_file(path, as_attachment=True)

if (__name__ == "__main__"):
     app.run(port = 5000)