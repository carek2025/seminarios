# 0. Settings
import cv2
import numpy as np

# 1. Cargar el modelo YOLO (Asegúrate de usar el modelo correcto)
from ultralytics import YOLO

# Cargar el modelo YOLOv11
model = YOLO('D:\\American Sign Language Letters.v1-v1.yolov11\\runs\\detect\\train2\\weights\\best.pt')  # Asegúrate de que la ruta sea correcta

# 2. Detección en tiempo real (Webcam)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Configurar la resolución de la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Realizar la detección
    results = model(frame)  # 'results' será una lista con los resultados

    # Si results es una lista, accedemos al primer elemento
    result = results[0]  # Tomamos el primer resultado de la lista

    # Renderizar los resultados (las detecciones)
    img = result.plot()  # Aquí usamos el método 'plot' para dibujar las cajas de detección

    # Mostrar el frame con las detecciones
    cv2.imshow('YOLO11s', img)

    # Terminar la captura si presionamos 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar la cámara y las ventanas
cap.release()
cv2.destroyAllWindows()
