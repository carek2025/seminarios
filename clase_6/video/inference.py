#importamos librerias
import torch
import cv2
import numpy as np
import pandas
#leeemos el modelo
model = torch.hub.load('ultralytics/yolov5','custom',
					   path = 'C:/carlos_enrique/seminarios/clase_6/video/model/cel-lapi-lent.pt')

# Ajustamos el umbral de confianza (opcional)
model.conf = 0.1  # Detecta más objetos al reducir el umbral
#Realisamos VideoCaptura
cap =cv2.VideoCapture(0)

#Empezamos:
while True:
	#Realizamos la lectura de la video captura
	ret,frame=cap.read()

	#Relaizamos las detecciones
	detect =model(frame)
    #mostramos informacion de la deteccion
	info=detect.pandas().xyxy[0]
	print(info)
	#Mostramos FPS
	cv2.imshow('Detector de lentes ,celular y lapicero',np.squeeze(detect.render()))

	#Leer el teclado
	t=cv2.waitKey(5)
	if t == 27:
		break;
cap.release()
cv2.destroyAllWindows()