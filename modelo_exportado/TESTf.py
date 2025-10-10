"""
SCRIPT 2: PRUEBA EN TIEMPO REAL
Archivo: probar_camara.py
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
from pathlib import Path
from collections import deque

class DetectorPostura:
    """
    Detecta la postura en tiempo real usando un modelo previamente entrenado
    """
    def __init__(self, ruta_modelo='modelo_exportado/modelo_postura.pkl', ventana=10):
        # Cargar modelo
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
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Ventana temporal para suavizar
        self.ventana = ventana
        self.buffer = deque(maxlen=ventana)  # guarda últimas N frames
    
    def extraer_caracteristicas_frame(self, frame):
        """Extrae características de pose de un frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        
        # Puntos clave
        cadera_izq = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        cadera_der = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        rodilla_izq = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        rodilla_der = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        tobillo_izq = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        tobillo_der = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        hombro_izq = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        hombro_der = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Características
        altura_cadera = (cadera_izq.y + cadera_der.y) / 2
        angulo_pierna_izq = self._calcular_angulo(
            [cadera_izq.x, cadera_izq.y],
            [rodilla_izq.x, rodilla_izq.y],
            [tobillo_izq.x, tobillo_izq.y]
        )
        angulo_pierna_der = self._calcular_angulo(
            [cadera_der.x, cadera_der.y],
            [rodilla_der.x, rodilla_der.y],
            [tobillo_der.x, tobillo_der.y]
        )
        separacion_pies = abs(tobillo_izq.x - tobillo_der.x)
        inclinacion_torso = abs(((hombro_izq.y + hombro_der.y)/2) - altura_cadera)
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
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def predecir(self, frame):
        """Predice la postura de un frame usando ventana temporal"""
        features = self.extraer_caracteristicas_frame(frame)
        if features is None:
            return None, None
        
        # Agregar al buffer
        self.buffer.append(features.flatten())
        
        if len(self.buffer) < self.ventana:
            return None, None  # no hay suficientes frames aún
        
        arr = np.array(self.buffer)  # shape: (ventana, 7)
        
        # Generar las 21 características como en entrenamiento
        caracteristicas_finales = np.concatenate([
            arr.mean(axis=0),
            arr.std(axis=0),
            arr.max(axis=0) - arr.min(axis=0)
        ]).reshape(1, -1)
        
        # Escalar y predecir
        features_scaled = self.scaler.transform(caracteristicas_finales)
        pred_idx = self.modelo.predict(features_scaled)[0]
        if hasattr(self.modelo, "predict_proba"):
            prob = self.modelo.predict_proba(features_scaled).max()
        else:
            prob = 1.0
        return self.clases[pred_idx], prob


# ============================================================================#
# SCRIPT PRINCIPAL
# ============================================================================#
if __name__ == "__main__":
    
    detector = DetectorPostura()
    cap = cv2.VideoCapture(0)  # Abrir cámara
    
    print("🎯 Detección de postura iniciada. Presiona 'q' para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ No se pudo capturar el frame")
            break
        
        clase, prob = detector.predecir(frame)
        
        # Mostrar resultado
        texto = f"{clase} ({prob*100:.1f}%)" if clase else "No detectado"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Detección de postura", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
