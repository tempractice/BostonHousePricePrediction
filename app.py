import pickle
import json
from flask import Flask,render_template,jsonify,app,request,url_for

import numpy as np
import pandas as pd

app=Flask(__name__)
#load the model
model=pickle.load(open('Boston_House_price_prediction.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    input_1=np.array(list(data.values()))
    input_1=np.delete(input_1,8)
    input_1=scaler.transform(input_1.reshape(1,-1))
    output=model.predict(input_1)

    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    data.pop(8)
    print(data)
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template('home.html', prediction_text='The predicted house price is {}'.format(output))



if __name__=='__main__':
    app.run(debug=True)
