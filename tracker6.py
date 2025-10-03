import cv2
import mediapipe as mp
import numpy as np
import socket
import time
import json
import joblib
from tensorflow.keras.models import load_model
import math

# --- UDP Config ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)



# --- Configuración modelo ---

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_video = mp_pose.Pose(static_image_mode=False,
                          model_complexity=1,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

WINDOW_SIZE = 30
KEYPOINTS = [
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]
CLASS_NAMES = ["Lento", "Rapido"]

model = load_model("velocity_lstm_lento_rapido.h5")
scaler = joblib.load("velocity_lstm_lento_rapido_scaler.pkl")
window = []

def calc_angle(a,b,c):
    a=np.array(a); b=np.array(b); c=np.array(c)
    ba = a-b; bc = c-b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle,-1.0,1.0)))

def extract_angles(frame_landmarks):
    coords = []
    for kp in KEYPOINTS:
        lm = frame_landmarks[kp.value]
        coords.append(np.array([lm.x, lm.y, lm.z]))

    L_HIP,R_HIP,L_KNEE,R_KNEE,L_ANKLE,R_ANKLE = coords
    hip_center = (L_HIP + R_HIP)/2
    L_HIP -= hip_center; R_HIP -= hip_center
    L_KNEE -= hip_center; R_KNEE -= hip_center
    L_ANKLE -= hip_center; R_ANKLE -= hip_center

    angles = [
        calc_angle(L_HIP,L_KNEE,L_ANKLE),
        calc_angle(R_HIP,R_KNEE,R_ANKLE),
        math.degrees(math.atan2(L_KNEE[1]-L_HIP[1], L_KNEE[0]-L_HIP[0])),
        math.degrees(math.atan2(R_KNEE[1]-R_HIP[1], R_KNEE[0]-R_HIP[0]))
    ]
    return angles

















# --- MediaPipe Pose ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_video = mp_pose.Pose(static_image_mode=False,
                          model_complexity=1,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# --- Global state ---
empezo = False
linea_hombros = None
pos_pies = None
last_steps = []
velocidad_actual = 0
ultimo_carril = "centro"
last_toggle_time = 0
cooldown = 2.0  # segundos

def detectPose(image, pose):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pose.process(rgb)

def checkHandsJoined(image, results):
    h, w, _ = image.shape
    lw = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * h)
    rw = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * w,
          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)
    dist = np.linalg.norm(np.array(lw) - np.array(rw))
    return "Hands Joined" if dist < 100 else "Hands Not Joined"

def checkLeftRight(image, results):
    h, w, _ = image.shape
    l_sh = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
    r_sh = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w)

    if (r_sh <= w//2 and l_sh <= w//2):
        return "izq"
    elif (r_sh >= w//2 and l_sh >= w//2):
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

def checkRunning(results, base_feet, frame):
    global velocidad_actual, window
    h, w, _ = frame.shape
    l_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * h
    r_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * h
    avg_feet = (l_foot + r_foot) / 2

    now = time.time()

    # --- Caso 1: pies no se levantaron ---
    if abs(avg_feet - base_feet) < 10:   # margen de 20 px
        # si en el último segundo no se movieron
        if len(window) == 0 or (now - last_steps[-1] if last_steps else 99) > 1.:
            velocidad_actual = 0
            return velocidad_actual

    # --- Caso 2: usar modelo ---
    angles = extract_angles(results.pose_landmarks.landmark)
    window.append(angles)

    if len(window) == WINDOW_SIZE:
        X_window = np.array(window).reshape(-1,4)
        X_window_scaled = scaler.transform(X_window).reshape(1, WINDOW_SIZE, 4)
        pred = model.predict(X_window_scaled, verbose=0)
        clase = CLASS_NAMES[np.argmax(pred)]
        window = []

        if clase == "Lento":
            velocidad_actual = 2
        elif clase == "Rapido":
            velocidad_actual = 3

    return velocidad_actual

# --- Main ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)

    results = detectPose(frame, pose_video)

    poscarril, poshorizontal, velocidad = "centro", "standing", 0
    hands = "Hands Not Joined"

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        hands = checkHandsJoined(frame, results)

        # --- toggle con cooldown ---
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
                print("[INIT] Línea hombros y pies guardados.")
            else:
                empezo = False
                linea_hombros, pos_pies = None, None
                velocidad_actual = 0
                print("[RESET] Tracking apagado, listo para nueva persona.")
            last_toggle_time = now

        # --- si está encendido, pero faltan pies u hombros ---
        if empezo:
            try:
                # validar pies
                l_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                r_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                if l_foot.visibility < 0.5 or r_foot.visibility < 0.5:
                    raise ValueError("pies no visibles")

                poscarril = checkLeftRight(frame, results)
                ultimo_carril = poscarril
                poshorizontal = checkJumpCrouch(frame, results, linea_hombros)

                if poshorizontal == "jumping":
                    velocidad = 3
                elif poshorizontal == "standing":
                    velocidad = checkRunning(results, pos_pies, frame)
                else:
                    velocidad = 0

                cv2.line(frame, (0, int(linea_hombros)), (w, int(linea_hombros)), (0, 255, 0), 5)

            except Exception as e:
                # tracking incompleto → valores neutros
                poscarril, poshorizontal, velocidad = "centro", "standing", 0
    else:
        # ❌ cuando no hay landmarks
        poscarril, poshorizontal, velocidad = "centro", "standing", 0


    # Overlay
    color = (0, 255, 0) if hands == "Hands Joined" else (0, 0, 255)
    cv2.putText(frame, f"HandsJoined: {hands}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"{poshorizontal}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"{poscarril}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Velocidad: {velocidad}", (w - 250, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Enviar JSON
    data = {"poscarril": poscarril, "poshorizontal": poshorizontal, "velocidad": velocidad}
    sock.sendto(json.dumps(data).encode(), (UDP_IP, UDP_PORT))

    cv2.imshow("bodytracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
