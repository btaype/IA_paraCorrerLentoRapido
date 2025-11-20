import cv2
import numpy as np
import mediapipe as mp
import pickle
import socket
import time
import json
import math
import warnings
from pathlib import Path
from collections import deque
import os

# --- IGNORAR WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================ #
#                           DETECTOR DE POSTURA MEJORADO                       #
# ============================================================================ #
class DetectorPostura:
    """
    Detecta la postura en tiempo real usando un modelo previamente entrenado
    Envía predicción cada 30 frames (1 segundo @ 30 FPS)
    """
    def __init__(self, ruta_modelo='modelo_exportado/modelo_postura.pkl', ventana=30):
        if not Path(ruta_modelo).exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {ruta_modelo}")
        
        with open(ruta_modelo, 'rb') as f:
            modelo_data = pickle.load(f)
        
        self.modelo = modelo_data['modelo']
        self.scaler = modelo_data['scaler']
        self.clases = modelo_data['clases']
        
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
            
        self.ventana = ventana  # 30 frames = 1 segundo
        self.buffer = deque(maxlen=ventana)
        self.frame_count = 0
        
    
    def extraer_caracteristicas_frame(self, frame):
        """Extrae 7 características de un frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        mp_pose = self.mp_pose
        
        cadera_izq = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        cadera_der = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        rodilla_izq = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        rodilla_der = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        tobillo_izq = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        tobillo_der = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        hombro_izq = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        hombro_der = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        altura_cadera = (cadera_izq.y + cadera_der.y) / 2
        angulo_pierna_izq = self._calcular_angulo([cadera_izq.x, cadera_izq.y], [rodilla_izq.x, rodilla_izq.y], [tobillo_izq.x, tobillo_izq.y])
        angulo_pierna_der = self._calcular_angulo([cadera_der.x, cadera_der.y], [rodilla_der.x, rodilla_der.y], [tobillo_der.x, tobillo_der.y])
        separacion_pies = abs(tobillo_izq.x - tobillo_der.x)
        inclinacion_torso = abs(((hombro_izq.y + hombro_der.y) / 2) - altura_cadera)
        movimiento_vertical = (tobillo_izq.visibility + tobillo_der.visibility) / 2
        diferencia_angular = abs(angulo_pierna_izq - angulo_pierna_der)
        
        return np.array([
            altura_cadera,
            angulo_pierna_izq,
            angulo_pierna_der,
            separacion_pies,
            inclinacion_torso,
            movimiento_vertical,
            diferencia_angular
        ])
    
    def _calcular_angulo(self, p1, p2, p3):
        """Calcula ángulo entre tres puntos"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    def predecir(self, frame):
        """
        Procesa frame y SIEMPRE predice cada 30 frames
        Retorna: (clase, probabilidad, debe_enviar)
        """
        features = self.extraer_caracteristicas_frame(frame)
        
        if features is None:
            self.frame_count += 1
            return None, None, False
        
        # Agregar al buffer
        self.buffer.append(features)
        self.frame_count += 1
        
        # PREDICCIÓN cada 30 frames (no espera a llenar buffer)
        debe_enviar = (self.frame_count % self.ventana == 0)
        
        if len(self.buffer) < self.ventana:
            # Aún acumulando, pero puedes devolver estado parcial
            return None, None, False
        
        if debe_enviar:
            # Calcular estadísticas sobre ventana de 30 frames
            arr = np.array(list(self.buffer))  # (30, 7)
            
            caracteristicas_finales = np.concatenate([
                arr.mean(axis=0),           # 7 features
                arr.std(axis=0),            # 7 features
                arr.max(axis=0) - arr.min(axis=0)  # 7 features
            ]).reshape(1, -1)  # (1, 21)
            
            # Normalizar con el scaler del entrenamiento
            features_scaled = self.scaler.transform(caracteristicas_finales)
            
            # Predecir
            pred_idx = self.modelo.predict(features_scaled)[0]
            
            if hasattr(self.modelo, "predict_proba"):
                prob = self.modelo.predict_proba(features_scaled).max()
            else:
                prob = 1.0
            
            clase_pred = self.clases[pred_idx]
            
            return clase_pred, prob, True
        
        return None, None, False


# ============================================================================ #
#                         TRACKER PRINCIPAL DEL JUEGO                          #
# ============================================================================ #

# --- UDP Config ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- CARGAR MODELO ---
print("Cargando modelo de postura...")
detector = DetectorPostura('modelo_exportado/modelo_postura.pkl', ventana=13)
print("✓ Modelo de postura cargado correctamente.")

# --- VARIABLES GLOBALES ---
empezo = False
linea_hombros = None
pos_pies = None
velocidad_actual = 0
ultimo_carril = "centro"
last_toggle_time = 0
cooldown = 2.0
ultima_clase = "Parado"  # Guardamos última predicción

# --- FUNCIONES AUXILIARES ---
def checkHandsJoined(image, results):
    h, w, _ = image.shape
    lw = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * h)
    rw = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * w,
          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)
    dist = np.linalg.norm(np.array(lw) - np.array(rw))
    return "Hands Joined" if dist < 20 else "Hands Not Joined"

def checkLeftRight(image, results):
    h, w, _ = image.shape
    l_sh = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
    r_sh = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w)
    if (r_sh <= w // 2 and l_sh <= w // 2):
        return "izq"
    elif (r_sh >= w // 2 and l_sh >= w // 2):
        return "der"
    else:
        return "centro"

def checkJumpCrouch(image, results, base_y):
    h, w, _ = image.shape
    l_sh = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
    r_sh = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
    mid_y = (l_sh + r_sh) // 2
    lower = base_y - 20
    upper = base_y + 70
    if mid_y < lower:
        return "jumping"
    elif mid_y > upper:
        return "crouching"
    else:
        return "standing"

# --- WEBCAM ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()
print("\n" + "=" * 60)
print("TRACKER MEJORADO - Predicción cada 30 frames (1 segundo)")
print("=" * 60)
print("Presiona ESC para salir\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    poscarril, poshorizontal, velocidad = "centro", "standing", 0
    hands = "Hands Not Joined"

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        hands = checkHandsJoined(frame, results)

        now = time.time()
        if hands == "Hands Joined" and (now - last_toggle_time > cooldown):
            if not empezo:
                empezo = True
                l_sh_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
                r_sh_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
                linea_hombros = (l_sh_y + r_sh_y) / 2
               
                print("[INIT] Tracking iniciado")
            else:
                empezo = False
                linea_hombros = None
                velocidad  = -3
                print("[RESET] Tracking reiniciado")
            last_toggle_time = now

        if empezo:
            poscarril = checkLeftRight(frame, results)
            ultimo_carril = poscarril
            poshorizontal = checkJumpCrouch(frame, results, linea_hombros)

            if poshorizontal == "jumping":
                velocidad = 2
            elif poshorizontal == "standing":
                # PREDICCIÓN cada 30 frames
                clase, prob, debe_enviar = detector.predecir(frame)
                
                if debe_enviar and clase is not None:
                    # Actualizar estado
                    ultima_clase = clase
                    print(f"🎯 PREDICCIÓN: {clase} (confianza: {prob:.2f})")
                    
                    if clase == "Parado":
                        velocidad = 0
                    elif clase == "Corriendo Nv1":
                        velocidad = 1
                    elif clase == "Corriendo Nv2":
                        velocidad = 3
                    else:
                        velocidad = velocidad_actual
                    velocidad_actual = velocidad
                else:
                    # Mantener velocidad anterior mientras acumula frames
                    velocidad = velocidad_actual
            else:
                velocidad = 0

            cv2.line(frame, (0, int(linea_hombros)), (w, int(linea_hombros)), (0, 255, 0), 3)

    fps = 1 / (time.time() - prev_time + 1e-6)
    prev_time = time.time()

    display_clase = "Parado" if velocidad == 0 else ("Velocidad 1" if velocidad == 1 else "Velocidad 2")
    color = (0, 255, 0) if hands == "Hands Joined" else (0, 0, 255)
    
    cv2.putText(frame, f"HandsJoined: {hands}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Buffer: {detector.frame_count % 13}/25", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"{poshorizontal}", (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"{poscarril}", (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Velocidad: {velocidad} ({display_clase})", (w - 500, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Enviar por UDP
    data = {"poscarril": poscarril, "poshorizontal": poshorizontal, "velocidad": velocidad}
    sock.sendto(json.dumps(data).encode(), (UDP_IP, UDP_PORT))

    cv2.imshow("bodytracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
print("✓ Sistema cerrado")