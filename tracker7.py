import cv2
import mediapipe as mp
import numpy as np
import socket
import time
import json
import joblib
from tensorflow.keras.models import load_model
import math
import warnings
import threading
import queue
from collections import deque
import os

# --- IGNORAR WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)

# --- UDP Config ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- CONFIGURACIÓN ---
WINDOW_SIZE = 60  # Siempre 60 frames como test2.py
KEYPOINTS = [
    mp.solutions.pose.PoseLandmark.LEFT_HIP,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
]
CLASS_NAMES = ["Lento", "Rápido"]

# Umbral de distancia entre tobillos para detectar "Parado"
ANKLE_DISTANCE_THRESHOLD = 0.11  # distancia normalizada (0-1)

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

def check_ankle_distance(landmarks):
    """Calcula la distancia entre tobillos para detectar posición parada"""
    l_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
    r_ankle = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
    
    distance = math.sqrt(
        (l_ankle.x - r_ankle.x)**2 + 
        (l_ankle.y - r_ankle.y)**2
    )
    
    return distance

# --- THREAD DE PREDICCIÓN CONTINUA (de test2.py) ---
class ContinuousPredictionThread(threading.Thread):
    def __init__(self, model, scaler):
        threading.Thread.__init__(self)
        self.model = model
        self.scaler = scaler
        self.window_queue = queue.Queue(maxsize=3)
        self.last_prediction = "Esperando..."
        self.running = True
        self.daemon = True
        self.lock = threading.Lock()
        
    def run(self):
        while self.running:
            try:
                window = self.window_queue.get(timeout=0.05)
                if window is not None:
                    X_window_scaled = self.scaler.transform(window).reshape(1, len(window), 4)
                    pred = self.model.predict(X_window_scaled, verbose=0)
                    
                    with self.lock:
                        self.last_prediction = CLASS_NAMES[np.argmax(pred)]
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en predicción: {e}")
    
    def add_window(self, window):
        """Añade ventana para predecir (descarta si cola llena)"""
        try:
            while self.window_queue.full():
                try:
                    self.window_queue.get_nowait()
                except:
                    break
            
            self.window_queue.put_nowait(window.copy())
        except queue.Full:
            pass
    
    def get_prediction(self):
        """Obtiene última predicción de forma thread-safe"""
        with self.lock:
            return self.last_prediction
    
    def stop(self):
        self.running = False

# --- FUNCIONES DE DETECCIÓN ---
def checkHandsJoined(image, results):
    h, w, _ = image.shape
    lw = (results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x * w,
          results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y * h)
    rw = (results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x * w,
          results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y * h)
    dist = np.linalg.norm(np.array(lw) - np.array(rw))
    return "Hands Joined" if dist < 100 else "Hands Not Joined"

def checkLeftRight(image, results):
    h, w, _ = image.shape
    l_sh = int(results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x * w)
    r_sh = int(results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x * w)

    if (r_sh <= w//2 and l_sh <= w//2):
        return "izq"
    elif (r_sh >= w//2 and l_sh >= w//2):
        return "der"
    else:
        return "centro"

def checkJumpCrouch(image, results, base_y):
    h, w, _ = image.shape
    l_sh = int(results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y * h)
    r_sh = int(results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y * h)
    mid_y = (l_sh + r_sh) // 2

    lower = base_y - 20
    upper = base_y + 70

    if mid_y < lower:
        return "jumping"
    elif mid_y > upper:
        return "crouching"
    else:
        return "standing"

# --- CARGAR MODELO ---
print("Cargando modelo...")
try:
    model = load_model("velocity_lstm_lento_rapido.h5", compile=False)
    
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        run_eagerly=False
    )
    
    scaler = joblib.load("velocity_lstm_lento_rapido_scaler.pkl")
    print("✓ Modelo cargado")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    exit()

# --- INICIALIZAR MEDIAPIPE ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # Lite model como test2.py
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# --- INICIAR THREAD DE PREDICCIÓN ---
prediction_thread = ContinuousPredictionThread(model, scaler)
prediction_thread.start()

# --- VARIABLES GLOBALES ---
empezo = False
linea_hombros = None
pos_pies = None
velocidad_actual = 0
ultimo_carril = "centro"
last_toggle_time = 0
cooldown = 2.0

# Usar deque para ventana deslizante (más eficiente)
window = deque(maxlen=WINDOW_SIZE)
frame_count = 0

# --- WEBCAM ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = time.time()
fps = 0

print("\n" + "="*50)
print("TRACKER OPTIMIZADO - PREDICCIÓN CONTINUA 60 FRAMES")
print("="*50)
print("Presiona ESC para salir\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_count += 1
    h, w, _ = frame.shape

    # Línea central
    cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)

    # Procesar pose
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True

    poscarril, poshorizontal, velocidad = "centro", "standing", 0
    hands = "Hands Not Joined"

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        hands = checkHandsJoined(frame, results)

        # --- Toggle con cooldown ---
        now = time.time()
        if hands == "Hands Joined" and (now - last_toggle_time > cooldown):
            if not empezo:
                empezo = True
                l_sh_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
                r_sh_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
                linea_hombros = (l_sh_y + r_sh_y) / 2
                l_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * h
                r_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * h
                pos_pies = (l_foot + r_foot) / 2
                print("[INIT] Tracking iniciado")
            else:
                empezo = False
                linea_hombros, pos_pies = None, None
                velocidad_actual = 0
                window.clear()
                print("[RESET] Tracking reiniciado")
            last_toggle_time = now

        # --- Si está encendido el tracking ---
        if empezo:
            try:
                # Validar visibilidad de pies
                l_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                r_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                if l_foot.visibility < 0.5 or r_foot.visibility < 0.5:
                    raise ValueError("pies no visibles")

                poscarril = checkLeftRight(frame, results)
                ultimo_carril = poscarril
                poshorizontal = checkJumpCrouch(frame, results, linea_hombros)

                # --- DETECCIÓN DE VELOCIDAD ---
                if poshorizontal == "jumping":
                    velocidad = velocidad_actual
                    
                elif poshorizontal == "standing":
                    # 🔴 PRIORIDAD 1: Verificar si está PARADO por distancia de tobillos
                    ankle_dist = check_ankle_distance(results.pose_landmarks.landmark)
                    
                    if ankle_dist > ANKLE_DISTANCE_THRESHOLD:
                        # ✅ PARADO: Piernas abiertas = velocidad 0 (sin usar modelo)
                        velocidad = 0
                        velocidad_actual = 0
                        window.clear()  # Limpiar ventana para nueva secuencia
                        last_clase = "Parado"  # Actualizar clase manualmente
                    else:
                        # 🔵 PRIORIDAD 2: Tobillos juntos = Usar modelo LSTM
                        angles = extract_angles(results.pose_landmarks.landmark)
                        window.append(angles)

                        # Cuando tenemos 60 frames, enviar a predecir
                        if len(window) == WINDOW_SIZE:
                            prediction_thread.add_window(np.array(window, dtype=np.float32))

                        # Obtener última predicción del modelo
                        last_clase = prediction_thread.get_prediction()
                        
                        # Mapear clase del modelo a velocidad
                        if last_clase == "Lento":
                            velocidad = 2
                        elif last_clase == "Rápido":
                            velocidad = 3
                        else:
                            # Esperando predicción inicial
                            velocidad = velocidad_actual
                        
                        velocidad_actual = velocidad
                else:
                    # crouching
                    velocidad = 0

                # Dibujar línea de hombros
                cv2.line(frame, (0, int(linea_hombros)), (w, int(linea_hombros)), (0, 255, 0), 3)

            except Exception as e:
                poscarril, poshorizontal, velocidad = "centro", "standing", 0

    # Calcular FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # --- OVERLAY ---
    # Obtener clase actual (puede ser del modelo o "Parado" manual)
    if empezo and velocidad == 0:
        display_clase = "Parado"
    else:
        display_clase = prediction_thread.get_prediction()
    color = (0, 255, 0) if hands == "Hands Joined" else (0, 0, 255)
    cv2.putText(frame, f"HandsJoined: {hands}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"{poshorizontal}", (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"{poscarril}", (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Velocidad con clase y buffer
    color_vel = (0, 255, 0) if velocidad == 0 else (0, 165, 255) if velocidad == 2 else (0, 0, 255)
    cv2.putText(frame, f"Velocidad: {velocidad} ({display_clase})", (w - 500, h - 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_vel, 2)
    cv2.putText(frame, f"Buffer: {len(window)}/60", (w - 300, h - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Enviar JSON por UDP
    data = {"poscarril": poscarril, "poshorizontal": poshorizontal, "velocidad": velocidad}
    sock.sendto(json.dumps(data).encode(), (UDP_IP, UDP_PORT))

    cv2.imshow("bodytracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# Limpieza
prediction_thread.stop()
prediction_thread.join(timeout=1)
cap.release()
cv2.destroyAllWindows()
pose.close()
print("✓ Sistema cerrado")