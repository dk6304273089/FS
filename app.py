from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
import pickle
import argparse
model = pickle.load(open('classifier.pkl', 'rb'))
app = Flask(__name__)

def predict():
    age=input("enter the value of age: ",type=NUMBER)
    sex=radio("CHOOSE THE GENDER: ",options=[0,1])
    cp=input("enter the cp: ",type=NUMBER)
    trestbps=input("enter the trestbps: ",type=NUMBER)
    chol=input("enter the chol: ",type=NUMBER)
    fbs=input("enter the fbs: ",type=NUMBER)
    restecg=input("enter the restecg: ",type=NUMBER)
    thalach=input("enter the thalach: ",type=NUMBER)
    exang=input("enter the exang: ",type=NUMBER)
    oldpeak=input("enter the value of oldpeak: ",type=FLOAT)
    slope=input("enter the value of slope: ",type=NUMBER)
    ca=input("enter the ca: ",type=NUMBER)
    thal=input("enter the value of thal: ",type=NUMBER)

    prediction=model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    output=prediction[0]
    put_text("the target value is",output)

app.add_url_rule('/tool', 'webio_view', webio_view(predict),
            methods=['GET', 'POST', 'OPTIONS'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(predict, port=args.port)
