
#import requests 
#from data_input import jobData

#URL = 'http://127.0.0.1:5000/predict'
#headers = {"Content-Type": "application/json"}
#data = {"input": jobData}

#r = requests.get(URL, headers=headers, json=data) 

#print(r.json())


import requests
from data_input import jobData, jData
from model import predic_index

url = 'http://localhost:5000/predict_api'
data = {"input": 1}
#data = predic_index(data["input"])

r = requests.post(url, json=data)


print(r.json())