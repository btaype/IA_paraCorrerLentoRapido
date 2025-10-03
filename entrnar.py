import pandas as pd
import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# --- CONFIG ---
WINDOW_SIZE = 60  # 2 segundos aprox si fps≈30

# --- Función para crear secuencias ---
def create_sequences_from_angles(df, label):
    X, y = [], []
    df = df.dropna()
    num_windows = len(df)//WINDOW_SIZE
    for i in range(num_windows):
        window = df.iloc[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE,:].values
        X.append(window)
        y.append(label)
    return X, y

# --- CARGAR CSVs ---
parado = pd.read_csv("parado.csv", header=None)
lento  = pd.read_csv("lento.csv", header=None)
rapido = pd.read_csv("rapido_recortado.csv", header=None)

Xp, yp = create_sequences_from_angles(parado, 0)
Xl, yl = create_sequences_from_angles(lento, 1)
Xr, yr = create_sequences_from_angles(rapido, 2)

X = np.array(Xp + Xl + Xr)
y = np.array(yp + yl + yr)

# --- NORMALIZAR ---
scaler = MinMaxScaler()
X_reshaped = X.reshape(-1,4)
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(X.shape)

# --- DIVIDIR TRAIN/TEST ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- MODELO LSTM ---
model = Sequential()
model.add(LSTM(128, input_shape=(WINDOW_SIZE,4), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- EARLY STOPPING ---
early_stop = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001, restore_best_weights=True)

# --- ENTRENAR HASTA ERROR MÍNIMO ---
model.fit(X_train, y_train, epochs=2000, batch_size=16,
          validation_data=(X_test, y_test),
          callbacks=[early_stop])

# --- GUARDAR MODELO Y SCALER ---
model.save("velocity_lstm_super.h5")
joblib.dump(scaler, "velocity_lstm_super_scaler.pkl")
print("¡Modelo entrenado con error mínimo y guardado!")