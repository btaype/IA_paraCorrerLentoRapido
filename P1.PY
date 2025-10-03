import socket
import json
import time

# Configuración del servidor (Unity escucha en este puerto)
UDP_IP = "127.0.0.1"  # localhost
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def enviar_estado(poscarril, poshorizontal, velocidad, repeticiones=50, delay=0.01):
    """
    Envía un estado como JSON a Unity 'repeticiones' veces, con un pequeño delay.
    """
    estado = {
        "poscarril": poscarril,       # "izq", "centro", "der"
        "poshorizontal": poshorizontal, # "jumping", "crouching", "" si nada
        "velocidad": velocidad       # 0..3
    }
    for i in range(repeticiones):
        msg = json.dumps(estado).encode("utf-8")
        sock.sendto(msg, (UDP_IP, UDP_PORT))
        print(f"Enviado ({i+1}/{repeticiones}):", estado)
        time.sleep(delay)  # muy pequeño delay para no saturar la red

# === Secuencia de prueba ===
time.sleep(3)  # esperar un segundo a que Unity esté listo
while True:
    enviar_estado("der", "jumping", 3)
    enviar_estado("centro", "", 1)
    enviar_estado("der", "", 3)
    
    enviar_estado("izq", "", 2)
    enviar_estado("centro", "crouching", 1)
    enviar_estado("centro", "", 0)
