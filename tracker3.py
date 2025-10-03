import cv2
import mediapipe as mp
import numpy as np
import socket
import time
import json

# --- UDP Config ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

# --- Funciones ---
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
        return "Left"
    elif (r_sh >= w//2 and l_sh >= w//2):
        return "Right"
    else:
        return "Center"

def checkJumpCrouch(image, results, base_y):
    h, w, _ = image.shape
    l_sh = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
    r_sh = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
    mid_y = (l_sh + r_sh) // 2

    lower = base_y - 20
    upper = base_y + 70

    if mid_y < lower:
        return "Jumping"
    elif mid_y > upper:
        return "Crouching"
    else:
        return "Standing"

def checkRunning(results, base_feet, frame):
    global last_steps, velocidad_actual
    h, w, _ = frame.shape
    l_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * h
    r_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * h
    avg_feet = (l_foot + r_foot) / 2

    now = time.time()

    # Detectar paso si se aleja del suelo
    if abs(avg_feet - base_feet) > 25:
        if len(last_steps) == 0 or (now - last_steps[-1]) > 0.25:
            last_steps.append(now)
            if len(last_steps) > 5:
                last_steps.pop(0)

            if len(last_steps) >= 2:
                avg_interval = (last_steps[-1] - last_steps[0]) / (len(last_steps)-1)
                if avg_interval > 0:
                    steps_per_sec = 1.0 / avg_interval
                    velocidad_actual = min(3, max(1, int(round(steps_per_sec))))  # 1–3
    else:
        # Si pasó demasiado tiempo sin pasos → resetear a 0
        if len(last_steps) > 0 and (now - last_steps[-1]) > 2.0:
            velocidad_actual = 0
            last_steps.clear()

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

    # Dibujar linea vertical fija
    cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)

    results = detectPose(frame, pose_video)

    # Valores por defecto
    poscarril, poshorizontal, velocidad = "Center", "Standing", 0
    hands = "Hands Not Joined"

    if results.pose_landmarks:
        # Dibujar skeleton completo
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        hands = checkHandsJoined(frame, results)
        poscarril = checkLeftRight(frame, results)

        if not empezo and hands == "Hands Joined":
            empezo = True
            l_sh_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
            r_sh_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
            linea_hombros = (l_sh_y + r_sh_y) / 2

            l_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * h
            r_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * h
            pos_pies = (l_foot + r_foot) / 2

            print("[INIT] Línea hombros y posición pies guardadas.")

        if empezo:
            poshorizontal = checkJumpCrouch(frame, results, linea_hombros)
            velocidad = checkRunning(results, pos_pies, frame)
            # Dibujar línea de hombros
            cv2.line(frame, (0, int(linea_hombros)), (w, int(linea_hombros)), (0, 255, 0), 2)

    # --- Overlay de textos ---
    # Superior izquierda: manos
    color = (0, 255, 0) if hands == "Hands Joined" else (0, 0, 255)
    cv2.putText(frame, f"HandsJoined: {hands}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Inferior izquierda: vertical + carril
    cv2.putText(frame, f"{poshorizontal}", (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"{poscarril}", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Inferior derecha: velocidad
    cv2.putText(frame, f"Velocidad: {velocidad}", (w - 250, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # --- Enviar JSON ---
    data = {
        "poscarril": poscarril,
        "poshorizontal": poshorizontal,
        "velocidad": velocidad
    }
    sock.sendto(json.dumps(data).encode(), (UDP_IP, UDP_PORT))

    cv2.imshow("bodytracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
