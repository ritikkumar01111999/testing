
import requests
from flask import request
import json
import tag_api
from tag_db import tagDb
import joblib
#from db import tagDb

def client(self):

    url = "http://127.0.0.1:5000/"
    testing = 'testing'
    testing_result = tag_api.model.predict(joblib.load('vectorizer.pkl').transform([testing]))
    r = requests.post(url, data = {'testing' : testing, 'testing_result' : testing_result})
    res = json.loads(r.text)
    tagDb().tag_db(testing,testing_result)