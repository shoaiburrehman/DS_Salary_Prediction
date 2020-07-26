import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
from model import predic_index, salary_estimate

app = Flask(__name__)

def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    ''' 
    int_features = [int(x) for x in request.form.values()]
    #int_features = int(x)  
    final_features = int(np.array(int_features))
    #print("int_features.value: ", int_features)
    #print("final_features: ", final_features)
    model = load_models()
    x_inp = np.array(predic_index(final_features)).reshape(1,-1)
    sal_est = salary_estimate(final_features)
    #print("x_inp: ", x_inp)
    #print("predic_index(final_features): ", predic_index(final_features))
    prediction = model.predict(x_inp)[0]
    output = round(prediction, 2)
    
    return render_template('index.html', sal_estimate= 'Salary from Glassdoor {}'.format(sal_est), prediction_text='Employee Predicted Salary is ${}K'.format(output))
   
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    request_json = request.get_json()
    x = request_json['input']
    
    print(x)
    x_in = np.array(predic_index(x)).reshape(1,-1)
    # load model
    model = load_models()
    prediction = model.predict(x_in)[0]
    response = json.dumps({'response': prediction})
    return response, 200


if __name__ == "__main__":
    app.run(debug=True)