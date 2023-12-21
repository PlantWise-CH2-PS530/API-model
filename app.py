from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import requests

from model import sc

app = Flask(__name__)

# Fungsi untuk memuat model yang sudah dilatih sebelumnya
def load_trained_model():
    model = load_model('model.h5')  # Ganti dengan path file model yang benar
    return model

# Fungsi untuk melakukan prediksi berdasarkan data cuaca
def predict_crop(latitude, longitude, start_date, end_date):
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
        print('HASIL WEATHER DATA : ', weather_data)

        hourly_temperature = [hourly['temperature_2m'] for hourly in weather_data.get('hourly', [{}])]
        hourly_humidity = [hourly['relative_humidity_2m'] for hourly in weather_data.get('hourly', [{}])]
        daily_rain = [daily['rain_sum'] for daily in weather_data.get('daily', [{}])]

        mean_temp = np.mean([temp for temp in hourly_temperature if temp is not None])
        mean_humidity = np.mean([hum for hum in hourly_humidity if hum is not None])
        mean_rain = np.mean([rain for rain in daily_rain if rain is not None])

        input_data = np.array([[mean_temp, mean_humidity, mean_rain]])

        model = load_trained_model()

        input_data = sc.transform(input_data)

        prediction = model.predict(input_data)

        predicted_indices = prediction[0].argsort()[-3:][::-1]
        predicted_labels = [str(index + 1) for index in predicted_indices]

        return predicted_labels
    else:
        return "Failed to fetch weather data"

# Endpoint API untuk prediksi tanaman
@app.route('/predict_crop', methods=['POST'])
def predict_crop_api():
    data = request.get_json()

    latitude = data.get('latitude')
    longitude = data.get('longitude')
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    if latitude is None or longitude is None or start_date is None or end_date is None:
        return jsonify({"error": "Invalid input. Please provide latitude, longitude, start_date, and end_date."}), 400

    prediction = predict_crop(latitude, longitude, start_date, end_date)
    return jsonify({"predicted_labels": prediction})

if __name__ == '__main__':
    app.run(debug=True)
