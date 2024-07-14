from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

Dst = pickle.load(open('Dst.pkl','rb'))
pp = pickle.load(open('PreP.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])

def predict():
    if request.method =='POST':
        Year = request.form['Year']
        avg_rainfall = request.form['avg_rainfall']
        Avg_Temp = request.form['Avg_Temp']
        State = request.form['State']
        District = request.form['District']
        Crop = request.form['Crop']

        feat = np.array([[Year, avg_rainfall, Avg_Temp, State, District, Crop]])
        NewFeat = pp.transform(feat)
        val = Dst.predict(NewFeat).reshape(1, -1)

        return render_template('index.html',val=val)

if __name__ =='__main__':
    app.run(debug=True)