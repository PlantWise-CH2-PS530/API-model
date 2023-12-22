import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import json
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix, classification_report

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from bs4 import BeautifulSoup
import requests


# Membuat fungsi API
def get_json(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return json.loads(response.content.decode('utf-8'))
    else:
        return None


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 52.52,
    "longitude": 13.41,
    "start_date": "2023-11-01",
    "end_date": "2023-12-01",
    "hourly": ["temperature_2m", "relative_humidity_2m"],
    "daily": "rain_sum"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°E {response.Longitude()}°N")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s"),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
), "temperature_2m": hourly_temperature_2m, "relative_humidity_2m": hourly_relative_humidity_2m}

hourly_dataframe = pd.DataFrame(data=hourly_data)
print(hourly_dataframe)

hourly_dataframe = hourly_dataframe.drop(['date'], axis=1)
hourly_dataframe.head()

# Menghitung rata-rata temperatur
mean_temp = hourly_dataframe['temperature_2m'].mean()

# Menghitung rata-rata kelembaban relatif
mean_humidity = hourly_dataframe['relative_humidity_2m'].mean()

# Menampilkan hasil
print(f"Temp: {mean_temp:.2f}")
print(f"Humidity: {mean_humidity:.3f}")

daily = response.Daily()
daily_rain_sum = daily.Variables(0).ValuesAsNumpy()

daily_data = {"date": pd.date_range(
    start=pd.to_datetime(daily.Time(), unit="s"),
    end=pd.to_datetime(daily.TimeEnd(), unit="s"),
    freq=pd.Timedelta(seconds=daily.Interval()),
    inclusive="left"
), "rain_sum": daily_rain_sum}

daily_dataframe = pd.DataFrame(data=daily_data)
print(daily_dataframe)

daily_dataframe= daily_dataframe.drop(['date'], axis=1)
daily_dataframe.head()

# Menghitung rata-rata temperatur
mean_rain = daily_dataframe['rain_sum'].mean()

# Menampilkan hasil
print(f"Rain: {mean_rain:.2f}")

df_crop = pd.read_csv('Fixed-Crop-Recommendation.csv')

df_crop = df_crop.drop(['N', 'P', 'K', 'ph'], axis=1)
df_crop.head()

X = df_crop.drop('label', axis=1)
y = df_crop['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

clf=RandomForestClassifier(n_estimators=60,max_depth=5,bootstrap=False,random_state=0)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:", accuracy_score(y_pred, y_test))

print(cross_val_score(clf,X,y,cv=10).mean())

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df_crop['label_encoded']= label_encoder.fit_transform(df_crop['label'])

X=df_crop.drop(['label','label_encoded'],axis=1)
y=df_crop['label_encoded']
print(len(y.unique()))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann=Sequential()
ann.add(keras.layers.Dense(128,input_shape=(3,),activation='relu'))
ann.add(keras.layers.Dense(64,input_shape=(3,),activation='relu'))
ann.add(keras.layers.Dense(28,input_shape=(3,),activation='relu'))
ann.add(keras.layers.Dense(22,input_shape=(3,),activation='softmax'))

ann.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.90:  # Ganti dengan akurasi yang diinginkan
            self.model.stop_training = True

# Callback untuk menghentikan pelatihan jika akurasi >= 0.95
custom_callback = CustomCallback()

ann.fit(x=X_train,y=y_train,batch_size=50,epochs = 300)

preds=ann.predict(X_test)

eval=ann.evaluate(X_test,y_test)
print(eval)

y_pred=[]
for x in preds:
    y_pred.append(np.argmax(x))
print(y_pred)

y_pred=np.array(y_pred)
print(y_pred.shape,y_test.shape)

from sklearn.metrics import multilabel_confusion_matrix,classification_report
print(y_pred.shape,y_test.shape)
multi_confuse_matrix=multilabel_confusion_matrix(y_test,y_pred)
accur=[]
for x in multi_confuse_matrix:
    accur.append((x[0][0])/(x[0][0]+x[1][1]+x[0][1]+x[1][0]))
print((sum(accur)/22)*100)

sensi=[]
for x in multi_confuse_matrix:
    sensi.append((x[0][0])/(x[0][0]+x[1][0]))
print("Sensitivity is ",(sum(sensi)/22)*100)
speci=[]
for x in multi_confuse_matrix:
    speci.append((x[1][1])/(x[1][1]+x[0][1]))
print("Specificity is ",(sum(speci)/22)*100)

labels=[f"{x}" for x in range(1,23)]
print(classification_report(y_test, y_pred, target_names=labels))

# Contoh input suhu, kelembaban, dan curah hujan
temperature = mean_temp  # input di dapatkan dari file csv nya
humidity = mean_humidity  # input di dapatkan dari file csv nya
rainfall = mean_rain  # input di dapatkan dari file csv nya

# Menyiapkan input untuk prediksi
input_data = np.array([[temperature, humidity, rainfall]])

# Normalisasi data input (jika Anda telah melakukan normalisasi saat pelatihan)
input_data = sc.transform(input_data)  # sc adalah objek StandardScaler yang digunakan saat pelatihan

# Melakukan prediksi dengan model
prediction = ann.predict(input_data)

# Mendapatkan indeks dari label dengan probabilitas tertinggi
predicted_indices = prediction[0].argsort()[-3:][::-1]

# Mendapatkan label kelas (nama tanaman) berdasarkan indeks hasil prediksi
predicted_labels = label_encoder.inverse_transform(predicted_indices)

# Menampilkan 3 hasil prediksi terbaik
print("Hasil prediksi terbaik:")
for i, label in enumerate(predicted_labels, start=1):
    print(f"Ranking-{i}: {label}")

ann.save('model.h5')

model = load_model('model.h5')