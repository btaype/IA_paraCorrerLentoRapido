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
from collections import deque

# --- IGNORAR WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURACIÓN ---
WINDOW_SIZE = 60  # Siempre 60 frames
KEYPOINTS = [
    mp.solutions.pose.PoseLandmark.LEFT_HIP,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
]
CLASS_NAMES = ["Lento", "Rápido"]

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

# --- THREAD DE PREDICCIÓN CONTINUA ---
class ContinuousPredictionThread(threading.Thread):
    def __init__(self, model, scaler):
        threading.Thread.__init__(self)
        self.model = model
        self.scaler = scaler
        self.window_queue = queue.Queue(maxsize=3)  # Cola de ventanas a procesar
        self.last_prediction = "Esperando..."
        self.running = True
        self.daemon = True
        self.lock = threading.Lock()
        
    def run(self):
        while self.running:
            try:
                window = self.window_queue.get(timeout=0.05)
                if window is not None:
                    # Procesar predicción
                    X_window_scaled = self.scaler.transform(window).reshape(1, len(window), 4)
                    pred = self.model.predict(X_window_scaled, verbose=0)
                    
                    # Actualizar resultado de forma thread-safe
                    with self.lock:
                        self.last_prediction = CLASS_NAMES[np.argmax(pred)]
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en predicción: {e}")
    
    def add_window(self, window):
        """Añade ventana para predecir (descarta si cola llena)"""
        try:
            # Si la cola está llena, descarta las viejas
            while self.window_queue.full():
                try:
                    self.window_queue.get_nowait()
                except:
                    break
            
            self.window_queue.put_nowait(window.copy())
        except queue.Full:
            pass  # Si está llena, simplemente ignora
    
    def get_prediction(self):
        """Obtiene última predicción de forma thread-safe"""
        with self.lock:
            return self.last_prediction
    
    def stop(self):
        self.running = False

# --- CARGAR MODELO ---
print("Cargando modelo...")
try:
    model = load_model("velocity_lstm_lento_rapido.h5", compile=False)
    
    # Optimización TensorFlow
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        run_eagerly=False
    )
    
    scaler = joblib.load("velocity_lstm_lento_rapido_scaler.pkl")
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
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# --- INICIAR THREAD DE PREDICCIÓN ---
prediction_thread = ContinuousPredictionThread(model, scaler)
prediction_thread.start()

# --- WEBCAM ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Usar deque para ventana deslizante (más eficiente)
window = deque(maxlen=WINDOW_SIZE)
frame_count = 0

import time
prev_time = time.time()
fps = 0

print("\n" + "="*50)
print("PREDICCIÓN CONTINUA CON 60 FRAMES")
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

        # CLAVE: Cuando tenemos 60 frames, enviamos a predecir
        # deque automáticamente mantiene solo los últimos 60
        if len(window) == WINDOW_SIZE:
            prediction_thread.add_window(np.array(window, dtype=np.float32))

        # Dibujar landmarks
        h, w, _ = frame.shape
        for i, kp in enumerate(KEYPOINTS):
            lm = results.pose_landmarks.landmark[kp.value]
            cx, cy = int(lm.x * w), int(lm.y * h)
            color = (0, 255, 0) if i < 2 else (255, 0, 255)
            cv2.circle(frame, (cx, cy), 4, color, -1)

    # Obtener última predicción
    last_clase = prediction_thread.get_prediction()

    # Calcular FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # Interfaz
    color_pred = (0, 165, 255) if last_clase == "Lento" else (0, 0, 255)
    cv2.putText(frame, last_clase, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_pred, 3)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Buffer: {len(window)}/60", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

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