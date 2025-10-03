import cv2
import time
import math
from ultralytics import YOLO

# Cargar modelo YOLOv8 preentrenado (pose estimation)
model = YOLO("yolov8n-pose.pt")  # también existe yolov8s-pose.pt (más preciso)

# Abrir cámara
cap = cv2.VideoCapture(0)

# Variables para velocidad
prev_time = time.time()
prev_x, prev_y = None, None
pixels_to_meters = 0.0025  # Ajustar con calibración real

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ejecutar detección de pose
    results = model(frame, verbose=False)

    for r in results:
        # Si encontró personas
        if r.keypoints is not None:
            # Tomar los keypoints (ejemplo: tobillo derecho = índice 16 en COCO)
            # COCO order: 0-nose, 1-eye, ... 15-ankle_left, 16-ankle_right
            kps = r.keypoints.xy.cpu().numpy()[0]  # primera persona
            x, y = kps[16]  # tobillo derecho

            cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)

            # Calcular distancia recorrida entre frames
            if prev_x is not None and prev_y is not None:
                dx = (x - prev_x) * pixels_to_meters
                dy = (y - prev_y) * pixels_to_meters
                dist = math.sqrt(dx**2 + dy**2)

                curr_time = time.time()
                dt = curr_time - prev_time
                speed = dist / dt  # m/s
                speed_kmh = speed * 3.6

                cv2.putText(frame, f"Velocidad: {speed_kmh:.2f} km/h",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2)

                prev_time = curr_time

            prev_x, prev_y = x, y

    cv2.imshow("YOLOv8 Pose - Velocidad", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
