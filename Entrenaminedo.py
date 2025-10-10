"""
SCRIPT 1: ENTRENAMIENTO
Este script entrena el modelo y exporta los pesos
Archivo: entrenar_modelo.py
"""

import cv2
import numpy as np
import os
from pathlib import Path
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import time
from tqdm import tqdm
import json

class EntrenadorPostura:
    """
    Entrena modelos de detección de postura y exporta los pesos
    """
    
    def __init__(self):
        self.modelos = {}
        self.scaler = StandardScaler()
        self.mejores_resultados = []
        
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.clases = {
            'parado': 0,
            'corriendo_nv1': 1,
            'corriendo_nv2': 2
        }
        self.nombres_clases = {0: 'Parado', 1: 'Corriendo Nv1', 2: 'Corriendo Nv2'}
    
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
        inclinacion_torso = abs(((hombro_izq.y + hombro_der.y) / 2) - altura_cadera)
        movimiento_vertical = (tobillo_izq.visibility + tobillo_der.visibility) / 2
        diferencia_angular = abs(angulo_pierna_izq - angulo_pierna_der)
        
        return [
            altura_cadera,
            angulo_pierna_izq,
            angulo_pierna_der,
            separacion_pies,
            inclinacion_torso,
            movimiento_vertical,
            diferencia_angular
        ]
    
    def _calcular_angulo(self, p1, p2, p3):
        """Calcula ángulo entre tres puntos"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def procesar_video(self, ruta_video, skip_frames=5):
        """Procesa un video y extrae características"""
        cap = cv2.VideoCapture(ruta_video)
        caracteristicas_video = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames == 0:
                features = self.extraer_caracteristicas_frame(frame)
                if features is not None:
                    caracteristicas_video.append(features)
            
            frame_count += 1
        
        cap.release()
        
        if len(caracteristicas_video) == 0:
            return None
        
        # Estadísticas
        caracteristicas_promedio = np.mean(caracteristicas_video, axis=0)
        caracteristicas_std = np.std(caracteristicas_video, axis=0)
        caracteristicas_max = np.max(caracteristicas_video, axis=0)
        caracteristicas_min = np.min(caracteristicas_video, axis=0)
        
        caracteristicas_finales = np.concatenate([
            caracteristicas_promedio,
            caracteristicas_std,
            caracteristicas_max - caracteristicas_min
        ])
        
        return caracteristicas_finales
    
    def cargar_dataset_desde_carpetas(self, ruta_dataset, skip_frames=5):
        """Carga todos los videos de las carpetas"""
        dataset_path = Path(ruta_dataset)
        
        if not dataset_path.exists():
            raise ValueError(f"La ruta {ruta_dataset} no existe")
        
        X = []
        y = []
        archivos_procesados = []
        
        print("=" * 60)
        print("CARGANDO DATASET DE VIDEOS")
        print("=" * 60)
        
        for carpeta_clase in dataset_path.iterdir():
            if not carpeta_clase.is_dir():
                continue
            
            nombre_clase = carpeta_clase.name.lower()
            
            if 'parado' in nombre_clase:
                etiqueta = 0
            elif 'nv1' in nombre_clase or 'nivel1' in nombre_clase or 'nivel_1' in nombre_clase:
                etiqueta = 1
            elif 'nv2' in nombre_clase or 'nivel2' in nombre_clase or 'nivel_2' in nombre_clase:
                etiqueta = 2
            else:
                print(f"⚠ Carpeta ignorada: {nombre_clase}")
                continue
            
            videos = list(carpeta_clase.glob('*.mp4')) + \
                     list(carpeta_clase.glob('*.avi')) + \
                     list(carpeta_clase.glob('*.mov'))
            
            print(f"\n📁 Procesando: {nombre_clase} (Clase {etiqueta})")
            print(f"   Videos encontrados: {len(videos)}")
            
            for video_path in tqdm(videos, desc=f"  Procesando"):
                caracteristicas = self.procesar_video(str(video_path), skip_frames)
                
                if caracteristicas is not None:
                    X.append(caracteristicas)
                    y.append(etiqueta)
                    archivos_procesados.append(str(video_path))
                else:
                    print(f"   ⚠ Error procesando: {video_path.name}")
        
        print(f"\n✓ Total de videos procesados: {len(X)}")
        print(f"  - Parado (0): {y.count(0)}")
        print(f"  - Corriendo Nv1 (1): {y.count(1)}")
        print(f"  - Corriendo Nv2 (2): {y.count(2)}")
        
        return np.array(X), np.array(y), archivos_procesados
    
    def entrenar_modelos(self, X_train, y_train):
        """Entrena múltiples modelos"""
        
        modelos_config = {
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=15, 
                                                   random_state=42, n_jobs=-1),
            'SVM_RBF': SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True),
            'SVM_Linear': SVC(kernel='linear', C=1, random_state=42, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'DecisionTree': DecisionTreeClassifier(max_depth=15, random_state=42),
            'NaiveBayes': GaussianNB()
        }
        
        print("\n" + "=" * 60)
        print("ENTRENANDO MODELOS")
        print("=" * 60)
        
        for nombre, modelo in modelos_config.items():
            inicio = time.time()
            modelo.fit(X_train, y_train)
            tiempo = time.time() - inicio
            
            self.modelos[nombre] = modelo
            print(f"✓ {nombre:20s} entrenado en {tiempo:.3f}s")
    
    def evaluar_modelos(self, X_test, y_test):
        """Evalúa todos los modelos"""
        
        print("\n" + "=" * 60)
        print("EVALUACIÓN DE MODELOS")
        print("=" * 60)
        
        resultados = []
        
        for nombre, modelo in self.modelos.items():
            y_pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            resultados.append({
                'modelo': nombre,
                'accuracy': acc
            })
            
            print(f"\n{nombre}:")
            print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        
        resultados = sorted(resultados, key=lambda x: x['accuracy'], reverse=True)
        self.mejores_resultados = resultados
        
        return resultados
    
    def mostrar_reporte_detallado(self, X_test, y_test, nombre_modelo=None):
        """Muestra reporte detallado"""
        
        if nombre_modelo is None:
            nombre_modelo = self.mejores_resultados[0]['modelo']
        
        print("\n" + "=" * 60)
        print(f"REPORTE DETALLADO: {nombre_modelo}")
        print("=" * 60)
        
        modelo = self.modelos[nombre_modelo]
        y_pred = modelo.predict(X_test)
        
        print("\nMatriz de Confusión:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Parado', 'Corriendo Nv1', 'Corriendo Nv2']))
    
    def exportar_modelo(self, carpeta_salida='modelo_exportado', nombre_modelo=None):
        """
        Exporta el modelo entrenado con toda la información necesaria
        """
        if nombre_modelo is None:
            nombre_modelo = self.mejores_resultados[0]['modelo']
        
        # Crear carpeta de salida
        Path(carpeta_salida).mkdir(exist_ok=True)
        
        modelo = self.modelos[nombre_modelo]
        
        # 1. Guardar modelo completo (pickle)
        modelo_completo = {
            'modelo': modelo,
            'scaler': self.scaler,
            'nombre': nombre_modelo,
            'clases': self.nombres_clases,
            'accuracy': self.mejores_resultados[0]['accuracy']
        }
        
        ruta_modelo = os.path.join(carpeta_salida, 'modelo_postura.pkl')
        with open(ruta_modelo, 'wb') as f:
            pickle.dump(modelo_completo, f)
        
        # 2. Guardar metadata (JSON)
        metadata = {
            'nombre_modelo': nombre_modelo,
            'accuracy': float(self.mejores_resultados[0]['accuracy']),
            'clases': self.nombres_clases,
            'fecha_entrenamiento': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_caracteristicas': 21,
            'ranking_modelos': [
                {'modelo': r['modelo'], 'accuracy': float(r['accuracy'])} 
                for r in self.mejores_resultados
            ]
        }
        
        ruta_metadata = os.path.join(carpeta_salida, 'metadata.json')
        with open(ruta_metadata, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print("\n" + "=" * 60)
        print("✓ MODELO EXPORTADO EXITOSAMENTE")
        print("=" * 60)
        print(f"📁 Carpeta: {carpeta_salida}/")
        print(f"📦 Modelo: modelo_postura.pkl")
        print(f"📋 Metadata: metadata.json")
        print(f"🎯 Accuracy: {metadata['accuracy']:.4f}")
        print(f"🤖 Algoritmo: {nombre_modelo}")
        print("=" * 60)
        
        return ruta_modelo, ruta_metadata


# ============================================================================
# SCRIPT PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

if __name__ == "__main__":
    
    print("🚀 INICIANDO ENTRENAMIENTO DE MODELO")
    print("=" * 60)
    
    # Configuración
    RUTA_DATASET = "dataset"  # ← Cambia esto por tu ruta
    SKIP_FRAMES = 5           # 5=rápido, 3=balance, 1=preciso
    CARPETA_SALIDA = "modelo_exportado"
    
    try:
        # 1. Crear entrenador
        entrenador = EntrenadorPostura()
        
        # 2. Cargar dataset
        X, y, archivos = entrenador.cargar_dataset_desde_carpetas(
            RUTA_DATASET, 
            skip_frames=SKIP_FRAMES
        )
        
        # 3. Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 4. Normalizar
        X_train = entrenador.scaler.fit_transform(X_train)
        X_test = entrenador.scaler.transform(X_test)
        
        # 5. Entrenar modelos
        entrenador.entrenar_modelos(X_train, y_train)
        
        # 6. Evaluar
        resultados = entrenador.evaluar_modelos(X_test, y_test)
        
        # 7. Ranking
        print("\n" + "=" * 60)
        print("RANKING DE MODELOS")
        print("=" * 60)
        for i, r in enumerate(resultados, 1):
            print(f"{i}. {r['modelo']:20s} - Accuracy: {r['accuracy']:.4f}")
        
        # 8. Reporte detallado
        entrenador.mostrar_reporte_detallado(X_test, y_test)
        
        # 9. EXPORTAR MODELO
        ruta_modelo, ruta_metadata = entrenador.exportar_modelo(
            carpeta_salida=CARPETA_SALIDA
        )
        
        print(f"\n✅ ¡Entrenamiento completado!")
        print(f"👉 Ahora ejecuta: python probar_camara.py")
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n📋 Estructura esperada:")
        print("dataset/")
        print("├── parado/")
        print("│   ├── video1.mp4")
        print("│   └── video2.mp4")
        print("├── corriendo_nv1/")
        print("│   └── video1.mp4")
        print("└── corriendo_nv2/")
        print("    └── video1.mp4")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()