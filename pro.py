# grabar_1min.py
import cv2
import time
from datetime import datetime

def main(output_filename=None, countdown=10, record_seconds=60, cam_index=0, fps=20):
    """
    Abre la cámara, muestra una cuenta regresiva (countdown) en pantalla
    durante `countdown` segundos, luego graba `record_seconds` segundos
    y guarda el vídeo en output_filename (si no se pasa, se genera con timestamp).
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("No se pudo abrir la cámara. Revisa el índice de cámara o permisos.")
        return

    # Intentar obtener tamaño de frame desde la cámara
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # Ajustar fps basado en cámara si está disponible
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps and cam_fps > 0:
        fps = int(cam_fps)

    if output_filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"grabacion_{ts}.mp4"

    # FOURCC para mp4 (puede variar según plataforma: 'mp4v' suele funcionar)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # ---------- Cuenta regresiva ----------
    start_count = time.time()
    while True:
        elapsed = int(time.time() - start_count)
        remaining = countdown - elapsed
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame de la cámara durante la cuenta regresiva.")
            break

        # Mostrar texto grande centrado con cuenta regresiva
        text = f"Grabando en: {remaining}s" if remaining >= 0 else "Iniciando..."
        # Si remaining <= 0, salimos del bucle
        cv2.putText(frame, text,
                    (int(width*0.05), int(height*0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 0), 4, cv2.LINE_AA)

        cv2.imshow("Pre-Grabacion (presiona 'q' para salir)", frame)

        if remaining <= 0:
            # pequeña pausa para que el usuario vea 0 antes de iniciar
            cv2.waitKey(500)
            break

        # permite abandonar con 'q'
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("Cancelado por el usuario durante la cuenta regresiva.")
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            return

    # ---------- Grabación ----------
    print(f"Iniciando grabación de {record_seconds} segundos...")
    record_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame de la cámara durante la grabación.")
            break

        # Escribir frame al archivo
        out.write(frame)

        # Mostrar tiempo restante en la ventana
        elapsed_rec = time.time() - record_start
        remaining_rec = int(record_seconds - elapsed_rec)
        timer_text = f"Tiempo: {remaining_rec}s"
        cv2.putText(frame, timer_text, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Grabando (presiona 'q' para detener)", frame)

        # detener si se alcanza el tiempo o si el usuario presiona 'q'
        if remaining_rec <= 0:
            print("Tiempo de grabación finalizado.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Grabación detenida por el usuario.")
            break

    # ---------- Liberar recursos ----------
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Grabación guardada en: {output_filename}")

if __name__ == "__main__":
    # Puedes cambiar los parámetros aquí si quieres:
    # output_filename, countdown (segundos), record_seconds, cam_index, fps
    main(output_filename=None, countdown=10, record_seconds=60, cam_index=0, fps=20)
