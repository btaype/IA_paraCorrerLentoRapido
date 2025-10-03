import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import math

# --- CONFIG ---
WINDOW_SIZE = 30  # 2 segundos aprox
KEYPOINTS = [
    mp.solutions.pose.PoseLandmark.LEFT_HIP,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
]
CLASS_NAMES = ["Lento", "Rápido"]

# --- Funciones ---
def calc_angle(a,b,c):
    a=np.array(a); b=np.array(b); c=np.array(c)
    ba = a-b
    bc = c-b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle,-1.0,1.0)))

def extract_angles(frame_landmarks):
    coords = []
    for kp in KEYPOINTS:
        lm = frame_landmarks[kp.value]
        coords.append(np.array([lm.x, lm.y, lm.z]))

    L_HIP,R_HIP,L_KNEE,R_KNEE,L_ANKLE,R_ANKLE = coords

    # Centrar cadera
    hip_center = (L_HIP + R_HIP)/2
    L_HIP -= hip_center
    R_HIP -= hip_center
    L_KNEE -= hip_center
    R_KNEE -= hip_center
    L_ANKLE -= hip_center
    R_ANKLE -= hip_center

    angles = [
        calc_angle(L_HIP,L_KNEE,L_ANKLE),
        calc_angle(R_HIP,R_KNEE,R_ANKLE),
        math.degrees(math.atan2(L_KNEE[1]-L_HIP[1], L_KNEE[0]-L_HIP[0])),
        math.degrees(math.atan2(R_KNEE[1]-R_HIP[1], R_KNEE[0]-R_HIP[0]))
    ]
    return angles

# --- Cargar modelo y scaler ---
model = load_model("velocity_lstm_lento_rapido.h5")
scaler = joblib.load("velocity_lstm_lento_rapido_scaler.pkl")

# --- Inicializar Mediapipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    smooth_landmarks=True, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# --- Captura webcam ---
cap = cv2.VideoCapture(0)
window = []
last_clase = "Esperando..."

print("Presiona 'q' para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        angles = extract_angles(results.pose_landmarks.landmark)
        window.append(angles)

        # Dibujar landmarks cadera hacia abajo
        h,w,_ = frame.shape
        for kp in KEYPOINTS:
            lm = results.pose_landmarks.landmark[kp.value]
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)

    # Si llenamos la ventana de 2 segundos
    if len(window) == WINDOW_SIZE:
        X_window = np.array(window).reshape(-1,4)
        X_window_scaled = scaler.transform(X_window).reshape(1, WINDOW_SIZE, 4)
        pred = model.predict(X_window_scaled, verbose=0)
        last_clase = CLASS_NAMES[np.argmax(pred)]  # actualizar predicción
        window = []  # limpiar ventana

    # Mostrar la última predicción siempre
    cv2.putText(frame, f"Prediccion: {last_clase}", (30,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

    cv2.imshow("Prediccion Velocidad", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
