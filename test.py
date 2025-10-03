import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import math
import warnings
import threading
import queue

# --- IGNORAR WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURACIÓN ---
WINDOW_SIZE = 60  # Reducido de 60 a 30
PREDICTION_INTERVAL = 10  # Reducido a cada 5 frames
KEYPOINTS = [
    mp.solutions.pose.PoseLandmark.LEFT_HIP,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
]
CLASS_NAMES = ["Parado", "Lento", "Rápido"]

# --- FUNCIONES ---
def calc_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def extract_angles(frame_landmarks):
    coords = np.array([[frame_landmarks[kp.value].x,
                        frame_landmarks[kp.value].y,
                        frame_landmarks[kp.value].z] for kp in KEYPOINTS], dtype=np.float32)
    
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = coords
    hip_center = (L_HIP + R_HIP) * 0.5
    coords -= hip_center
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = coords
    
    return np.array([
        calc_angle(L_HIP, L_KNEE, L_ANKLE),
        calc_angle(R_HIP, R_KNEE, R_ANKLE),
        math.degrees(math.atan2(L_KNEE[1] - L_HIP[1], L_KNEE[0] - L_HIP[0])),
        math.degrees(math.atan2(R_KNEE[1] - R_HIP[1], R_KNEE[0] - R_HIP[0]))
    ], dtype=np.float32)

# --- THREAD DE PREDICCIÓN (CLAVE PARA VELOCIDAD) ---
class PredictionThread(threading.Thread):
    def __init__(self, model, scaler):
        threading.Thread.__init__(self)
        self.model = model
        self.scaler = scaler
        self.queue = queue.Queue(maxsize=2)  # Solo 2 elementos en cola
        self.result_queue = queue.Queue(maxsize=1)
        self.running = True
        self.daemon = True
        
    def run(self):
        while self.running:
            try:
                window = self.queue.get(timeout=0.1)
                if window is not None:
                    X_window_scaled = self.scaler.transform(window).reshape(1, len(window), 4)
                    pred = self.model.predict(X_window_scaled, verbose=0)
                    
                    # Limpiar cola de resultados y poner nuevo
                    while not self.result_queue.empty():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    self.result_queue.put(CLASS_NAMES[np.argmax(pred)])
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en predicción: {e}")
    
    def predict(self, window):
        # Limpiar cola si está llena
        while self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        
        try:
            self.queue.put_nowait(window.copy())
        except queue.Full:
            pass
    
    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        self.running = False

# --- CARGAR MODELO ---
print("Cargando modelo...")
try:
    model = load_model("velocity_lstm_super.h5", compile=False)
    
    # Optimización TensorFlow
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        run_eagerly=False
    )
    
    scaler = joblib.load("velocity_lstm_super_scaler.pkl")
    print("✓ Modelo cargado")
except Exception as e:
    print(f"Error: {e}")
    exit()

# --- INICIALIZAR MEDIAPIPE ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # Lite model
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.3,  # Más permisivo
    min_tracking_confidence=0.3
)

# --- INICIAR THREAD DE PREDICCIÓN ---
prediction_thread = PredictionThread(model, scaler)
prediction_thread.start()

# --- WEBCAM ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Más pequeño
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo

window = []
last_clase = "Esperando..."
frame_count = 0

import time
prev_time = time.time()
fps = 0

print("\n" + "="*50)
print("SISTEMA ULTRA-RÁPIDO INICIADO")
print("="*50)
print("Presiona 'q' para salir\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # Procesar pose
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True

    if results.pose_landmarks:
        angles = extract_angles(results.pose_landmarks.landmark)
        window.append(angles)

        if len(window) > WINDOW_SIZE:
            window.pop(0)

        # Enviar a predicción en thread separado
        if frame_count % PREDICTION_INTERVAL == 0 and len(window) == WINDOW_SIZE:
            prediction_thread.predict(np.array(window, dtype=np.float32))

        # Obtener resultado si está disponible (no bloquea)
        result = prediction_thread.get_result()
        if result:
            last_clase = result

        # Dibujar SOLO landmarks (sin líneas para mayor velocidad)
        h, w, _ = frame.shape
        for i, kp in enumerate(KEYPOINTS):
            lm = results.pose_landmarks.landmark[kp.value]
            cx, cy = int(lm.x * w), int(lm.y * h)
            color = (0, 255, 0) if i < 2 else (255, 0, 255)  # Verde cadera, magenta resto
            cv2.circle(frame, (cx, cy), 4, color, -1)

    # Calcular FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # Interfaz minimalista
    color_pred = (0, 255, 0) if last_clase == "Parado" else (0, 165, 255) if last_clase == "Lento" else (0, 0, 255)
    cv2.putText(frame, last_clase, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_pred, 3)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Velocidad", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
prediction_thread.stop()
prediction_thread.join(timeout=1)
cap.release()
cv2.destroyAllWindows()
pose.close()
print("✓ Sistema cerrado")