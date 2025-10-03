import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import time

# --- CONFIG ---
FPS = 30
DURATION = 120  # segundos por clase
KEYPOINTS = [
    mp.solutions.pose.PoseLandmark.LEFT_HIP,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
]

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

    # Calcular ángulos
    angles = [
        calc_angle(L_HIP,L_KNEE,L_ANKLE),
        calc_angle(R_HIP,R_KNEE,R_ANKLE),
        math.degrees(math.atan2(L_KNEE[1]-L_HIP[1], L_KNEE[0]-L_HIP[0])),
        math.degrees(math.atan2(R_KNEE[1]-R_HIP[1], R_KNEE[0]-R_HIP[0]))
    ]
    return angles

# --- Grabación con visualización ---
def record_class(class_name):
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        smooth_landmarks=True, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    all_angles = []

    # Cuenta regresiva
    for i in range(10,0,-1):
        ret, frame = cap.read()
        cv2.putText(frame, f"Comienza en {i} s", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        cv2.imshow(f"Grabando {class_name}", frame)
        cv2.waitKey(1000)
    start_time = time.time()

    print(f"Grabando clase '{class_name}'...")

    while time.time() - start_time < DURATION:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            # Extraer ángulos
            angles = extract_angles(results.pose_landmarks.landmark)
            all_angles.append(angles)

            # Dibujar landmarks cadera hacia abajo
            for kp in KEYPOINTS:
                lm = results.pose_landmarks.landmark[kp.value]
                h,w,_ = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)

        cv2.putText(frame, f"Grabando {class_name}", (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow(f"Grabando {class_name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Guardar CSV
    df = pd.DataFrame(all_angles)
    df.to_csv(f"{class_name}.csv", index=False, header=False)
    print(f"Guardado {len(all_angles)} frames en {class_name}.csv")
    cap.release()
    cv2.destroyAllWindows()

# --- Uso ---
record_class("parado")
record_class("lento")
record_class("rapido")
