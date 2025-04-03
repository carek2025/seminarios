Readme:

Estructura:
clase_6/
│── Algoritmo-de-vision-computacional/
│   ├── .idea/  # Configuración del entorno de desarrollo (PyCharm)
│   ├── data/   # Carpeta de datos para entrenamiento y validación
│   │   ├── images/
│   │   │   ├── train/  # Imágenes de entrenamiento
│   │   │   ├── val/    # Imágenes de validación
│   │   ├── labels/
│   │   │   ├── train/  # Etiquetas de entrenamiento marcadas con make sense
│   │   │   ├── val/    # Etiquetas de validación marcadas por make sense
│   │   │   ├── train.cache  # Archivos de caché generados al entrenarlo
│   │   │   ├── train.cache.npy
│   │   │   ├── val.cache
│   │   │   ├── val.cache.npy
│   ├── yolov5/  # Directorio donde está el código de YOLOv5 se descarga desde GitHub
│   │   │   ├── data/
│   │   │   │   ├── coco128.yaml
│   │   │   ├── venv/
│   │   │   │   ├── Scripts/
│   │   │   │   │   ├── python.exe
│   ├── train_yolo.py  # Script de entrenamiento
│── External Libraries/  # Librerías externas utilizadas en el entorno
│── Scratches and Consoles/  # Consolas y borradores en el entorno de desarrollo


--Crear tu projecto: mkdir Algoritmo-de-vision-computaciona
--Entrar a tu projecto: cd Algoritmo-de-vision-computaciona
--clonar el repo de yolov5: git clone https://github.com/ultralytics/yolov5.git
--Crear un entorno virtual:
----cd yolov5
----python -m venv venv
----venv\Scripts\activate
--instalar los paquetes necesarios:
----pip install -r requirements.txt
----pip install roboflow
--Si te sale un error intenta instalar versiones compatibles:
----pip install --upgrade protobuf==5.26.1
----pip install --upgrade pillow==10.2.0
----pip install --upgrade grpcio-tools==1.71.0
----pip install --upgrade moviepy==2.1.1
--Luego vuelve a intentar a instalar las dependencias:
----pip install -r requirements.txt
----pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
----pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
--Probar el pythorch:
----python -c "import torch; print(torch.__version__)"
--Crear tu dataset:
----Definir que clases o categorías o de que tipos de objetos haras: por ejemplo en esta ocacion haremos Lentes, Lapicero y Celular
----Descargar imágenes para las clases:
------En esta ocacion usaremos la herramienta 'Download All Images' para descargar un conjunto de imágenes buscando por la web ,pasos:buscar en la web la herramienta y descargar la extencion de Google, buscar por la web la clase, de los resultados ir a al categoría de imágenes, cargar tantas imágenes como quieras descargar bajando el mouse, y da click en la extencion(al activarla descargar tantas imágenes como cargaste), descargar todos y guardarlo en una carpeta que crearas en tu projecto con el nombre de la clase(esta carpeta solo sera temporal ya que solo lo usaremos para sleccionar las images que sirven y aparece el objeto en esa imagen,si no aparece o no carga el formato lo eliminamos la imagen,para hacer mas facil el filtrado lo ponemos por tipo para que se juntes los formatos que no sirven y asi eliminar mas facil)-->estos paso se repiten tantas veces como clases tengas
------crear las carpeta data y las subcarpetas dentro de ella:
mkdir data
cd data
mkdir images
mkdir labels
cd images
mkdir train
mkdir val
cd ../labels
mkdir train
mkdir val
------Una vez filtrado las imagenes y creado las carpetas, separaremos 70 - 30 las imagenes de la clase poniendolo en images/train y images/val -->este se repite tantas clases como tengas(todas las imagenes de todas las clases iran en esas carpetas pero en diferentes proporciones para que una parte sirva para entrenar y otra para validar)
------Continuaremos y empezaremos el proceso de etiquetado con la herramienta 'make sense', lo buscamos en la web y damos click al primer enlace ,damos click en get started dentro de la web,y movemos toda la carpeta de images/train al cuadro que aparece en pantalla que tiene las palabras "Drop images",una ves cargado selecciones la opcion 'Object Detection',nos aparecera un cuadro donde crearemos las clases para etiquetar(en las parte izquierda superior del cuadro aparece un signo de '+' que es para agregar otra etiqueta o clase, una vez aplastado se creo un input donde puedes colocar tu nombre del label y puedes selecciopnar el color que se pintara al seleccionar despues,este paso lo repetiras tanatas veces como clases tengas o puedes agregar un archivo de texto donde tienes los labels escritos),una vez completa el proceso de creacion de labels se aplastara en la opcion 'Start project',te aparecera una interfaz con un seleccionador de areas(sirve y se utiliza para seleccionar la zona exacta donde esta el objeto de la clase),una vez seleccionado el area en la parte derecha sleecciona el label en el iput que dice 'select label',una ves seleccionado el label ya habras yterminado con esa imagen en la parte izquierda selecciona otar imagen para que etiquetes(Nota:Una vez que seleccionas un label en una imagen es seleccion de labels se queda como prewdeterminado cada vez que selecciones un area de las imagenes siguientes asi que trata de etiquetar una clase de imagenes primera y de ahi pasas a las siguienets clases),una ves terminado de etiquetar tus imagenes seleccioanla opcion de 'Actions' y de esa opcion seleccionas la opcion 'Export Annotations' y como en esta ocacion lo haremos con yolo seleccionas el formato .zip de yolo ,lo descomprimes y lo pones en labels/train todos los labels que son de extencion .txt --> este paso se repite con images/val y lo pones en labels/val
--creamos un archivo .yaml en  yolov5/data/scripts/ que sera esquema como coco.yaml, en esta ocacion mi archivo se llamara custom.yaml  :
cd Algoritmo-de-vision-computaciona/yolov5/data/scripts
nano custom.yaml
contenido pero lo cambias el path por la ruta en tu maquina:
path: "C:\\carlos_enrique\\seminarios\\clase_6\\Algoritmo-de-vision-computacional\\data"
train: images\\train
val: images\\val
test:

names:
  0: Celular
  1: Lentes
  2: Lapicero


# Download script/URL (optional)
download: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
 --Seleccionamos al interpreter de yolov5/venv/Scripts para que lo use el IDE con 'ctrl+shift+p' y se nos abrirá un input con opciones ,ponemos en el input Python interpreter y damos click a la primera opción seleccionamos add new interpreter y buscamos una opción para agregar una url de un interpreter local ,cuadno o encontramos copiamos el path absoluto de yolov5/venv/Scripts/python.exe y lo ponemos en esa opción o si nos deja seleccionar mediante el sistemas archivos entonces te vas al python.exe y le das click.
--creamos un archivo.py en la raíz del proyecto que llegaría ser la carpeta 'Algoritmo-de-vision-computaciona',nos movemos ahí mediante cd 'nombre-carpeta' o cd ..(retroceder un carpeta) y lo creamos:
----nano train_yolo.py
Nota:  primero revisar la siguiente explicacion para que cambies los parametros del comando que ejecutaras en el entrenamiento
Este comando es una lista de argumentos que se le pasa al intérprete de Python para ejecutar el script de entrenamiento de YOLOv5. Cada elemento en la lista tiene un propósito específico en la configuración del entrenamiento. A continuación, se detalla cada parte:

"python": Invoca el intérprete de Python para ejecutar el script.

r"C:\carlos_enrique\seminarios\clase_6\Algoritmo-de-vision-computacional\yolov5\train.py": Ruta absoluta al script de entrenamiento de YOLOv5.

"--img", "566": Especifica el tamaño de las imágenes de entrada (en píxeles) que se utilizarán para el entrenamiento.

"--batch", "6": Define el tamaño del lote (batch) a procesar en cada iteración durante el entrenamiento.

"--epochs", "70": Indica la cantidad de épocas (ciclos completos sobre el dataset) que se ejecutarán durante el entrenamiento.

"--data", r"C:\carlos_enrique\seminarios\clase_6\Algoritmo-de-vision-computacional\yolov5\data\custom.yaml": Ruta al archivo de configuración YAML que contiene los paths y la definición de las clases para el dataset.

"--weights", "yolov5x.pt": Especifica el archivo de pesos pre-entrenados que se usará para iniciar el entrenamiento (por ejemplo, para aplicar transfer learning),se selecciona de la pagina de git de los modelos de yolov5.

"--cache": Habilita el uso de caché para acelerar el entrenamiento, almacenando temporalmente datos procesados.

Este comando se utiliza dentro de un proceso (por ejemplo, a través de subprocess.Popen) para ejecutar el entrenamiento del modelo de detección de objetos configurado según los parámetros anteriores.

contenido pero lo cambias los path que son para esos archivos pero de tu maquina:
import os
import subprocess

command = [
    "python", r"C:\carlos_enrique\seminarios\clase_6\Algoritmo-de-vision-computacional\yolov5\train.py",
    "--img", "566",
    "--batch", "6",
    "--epochs", "70",
    "--data", r"C:\carlos_enrique\seminarios\clase_6\Algoritmo-de-vision-computacional\yolov5\data\custom.yaml",
    "--weights", "yolov5x.pt",
    "--cache"
]

# Ejecutar el proceso con stdout y stderr visibles
process = subprocess.Popen(
    command,
    cwd=r"C:\carlos_enrique\seminarios\clase_6\Algoritmo-de-vision-computacional\yolov5",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True,
    encoding="utf-8"  # ⬅ Agrega esta línea
)


# Leer la salida en tiempo real e imprimirla
for line in process.stdout:
    print(line, end='')

# Esperar a que termine
process.wait()
--Le das a ejecutar el archivo en tu IDE y te aparece una salida que se esta entrenando
--una ves entrenado tu modelo0 se guardaraa tu modelo entrenad en la carpeta yolov5/runs/train dentro de esa carpeta se genera una carpeta con las iniciales expX de ahí el x mayor es tu carpeta que recién se entreno y dentro de esa carpeta estan todos los archivos que genera al entrenar el modelo yolov5 ,dentro de la carpeta exp están los pesos o weights y ahí esta tu modelo entrenado ,normalmente es el archivo best.pt.
--------------------Proceso de la inferencia o prueba --------------------------------------------------
creamos un projecto en pychar con variables de entorno creadas también que tenga de interprete a python 8
cremos una carpeta model dentro:mkdir model
nos movemos dentro: cd model
copiamos nuestro modelo entrenado llamado best.pt y lo pegamos aca, también el custom.yaml
nos salimos fuera de la carpeta y cremos un archivo de Python:nano inference.py
dentro ponemos el siguiente contenido pero se le cambian los path del modelo entrenado con el tuyo:
#importamos librerias
import torch
import cv2
import numpy as np
import pandas
#leeemos el modelo
model = torch.hub.load('ultralytics/yolov5','custom',
					   path = 'C:/carlos_enrique/video/model/cel-lapi-lent.pt')

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

Haces correr al modelo y ya te funcionara

Nota si te muevas o atra carpeta o por cualquier cosa algo del interpretese mueve lo cambias al interorete de tu venv creado,le das 'ctrl+shift+p',escribes select interpreter y le das a cambiar el interprete o la primera opción que te salga y selecciones el 3.8 pero si no parece creas uno nuevo con las opciones que tienes abajo,le das crear nuevo interpretesr y te dará mas opciones como generar nuevo o sleecinar uno que tienes ,seleccionamos el interpreter que se creo con nuestro proyecto seleccionando la ruta de dentro del venv/Scripts/Python.exe de nuestro proyecto creado anteriormente

---------------------------------------Extra(comandos que quisa te sirvan)-------------------------------------
taskkill /F /IM python.exe /T
taskkill /F /IM pythonw.exe /T
taskkill /F /IM python.exe /IM pythonw.exe
wmic process where "name='python.exe'" delete
wmic process where "name='pythonw.exe'" delete
Stop-Process -Name python -Force
Stop-Process -Name pythonw -Force

---------

pip install --upgrade protobuf==5.26.1
pip install --upgrade pillow==10.2.0
pip install --upgrade grpcio-tools==1.71.0
pip install --upgrade moviepy==2.1.1
pip install idna==3.7

#A veces, estas dependencias cambian. Para comprobar si roboflow sigue necesitando idna==3.7, ejecuta:
pip show roboflow

pip install -r requirements.txt
pip install --force-reinstall -r requirements.txt

pip list | findstr torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu