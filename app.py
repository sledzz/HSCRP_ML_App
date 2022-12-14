from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow import keras
import tensorflow_addons as tfa
import joblib

app = Flask(__name__)
scaler = joblib.load('scaler.save')
model = keras.models.load_model('focal_model.h5', compile=False)

def get_bmi(weight, height):
    kg = float(weight)*0.45359237
    m = float(height)*0.0254
    return float(kg/(m**2))

def get_sleep_cat(hour):
    print(hour)
    if 9 <= hour <= 9.5: return 1
    elif  9.5 < hour <= 10.5: return 2
    elif 10.5 < hour <= 12.5: return 3
    elif 1 <= hour <=2: return 4
    else: return 5
    
def get_wakeup_cat (hour):
    if  4 <= hour <=5 : return 1
    elif 5 < hour <=6.5 : return 2
    elif 6.5 <= hour <8: return 3
    elif 8 <= hour <=9.5: return 4
    else: return 5

def get_sex(sex):
    if sex == "Male": return 0
    elif sex == 'Female': return 1
    else: return 'Please enter a valid sex: Male or Female'

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    raw_features  = request.form
    features = {
        'Gender' : get_sex(raw_features['Gender']),
        'Age': float(raw_features['Age']),
        'BMI' : float(get_bmi(weight=raw_features['Weight'], height =raw_features['Height'])),
        'Systolic': float(raw_features['Systolic']),
        'Diastolic' : float(raw_features['Diastolic']),
        'Alcohol': float(raw_features['Alcohol']),
        'Sleep_hrs' : float(raw_features['Sleep_hrs']),
        'SleepTimeCat': get_sleep_cat(float(raw_features['SleepTimeCat'])),
        'WakeUpCat' : get_wakeup_cat(float(raw_features['WakeUpCat']))
        }

    model.compile(optimizer=keras.optimizers.Adam(0.0005), loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.675, gamma=4))

    features_array = np.reshape(np.array(list(features.values())), (1, -1))

    transformed_features = scaler.transform(features_array)

    try: 
        result = round(float(model.predict(transformed_features).flatten()[0]), 3)*100
    except TypeError as err:
        return jsonify({'error': str(err)})
         

    return render_template('index.html', prediction_text =  f"The probability you have high levels of HSCRP is: {result}%")
        

if __name__ == '__main__':
    app.run(debug='True')