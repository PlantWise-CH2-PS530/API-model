from flask import Flask, request, jsonify
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

from model import label_encoder, X_train

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')
sc = StandardScaler()
sc.fit(X_train)

# Function to fetch data from meteo API
def get_meteo_data(latitude, longitude, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m"],
        "daily": "rain_sum"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        weather_data = response.json()

        # Get hourly data
        hourly_data = weather_data['hourly']
        temperature = np.mean(hourly_data['temperature_2m'])
        humidity = np.mean(hourly_data['relative_humidity_2m'])

        # Get daily data
        daily_data = weather_data['daily']
        rainfall = np.mean(daily_data['rain_sum'])

        return {
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall
        }
    else:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Get user input
    latitude = data['latitude']
    longitude = data['longitude']
    start_date = data['start_date']
    end_date = data['end_date']

    # Get meteo data
    meteo_data = get_meteo_data(latitude, longitude, start_date, end_date)
    temperature = meteo_data['temperature']
    humidity = meteo_data['humidity']
    rainfall = meteo_data['rainfall']

    # Prepare input data for prediction
    input_data = np.array([[temperature, humidity, rainfall]])

    # Normalize input data
    input_data = sc.transform(input_data)

    # Perform prediction
    prediction = model.predict(input_data)

    # Get the top 3 predicted labels
    predicted_indices = prediction[0].argsort()[-3:][::-1]
    predicted_labels = label_encoder.inverse_transform(predicted_indices)

    # Return prediction result
    result = {
        "predictions": [
            {"rank": i + 1, "label": label} for i, label in enumerate(predicted_labels)
        ]
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)