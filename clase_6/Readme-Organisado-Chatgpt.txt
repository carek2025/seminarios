A continuación se presenta el README reorganizado de forma lineal y estructurada. Se ha mantenido todo el contenido original, únicamente se ha reordenado y se ha dado una estructura más clara para facilitar su lectura e implementación:

---

# Readme

## 1. Estructura del Proyecto

```
clase_6/
│── Algoritmo-de-vision-computacional/
│   ├── .idea/                     # Configuración del entorno de desarrollo (PyCharm)
│   ├── data/                      # Carpeta de datos para entrenamiento y validación
│   │   ├── images/
│   │   │   ├── train/             # Imágenes de entrenamiento
│   │   │   ├── val/               # Imágenes de validación
│   │   ├── labels/
│   │   │   ├── train/             # Etiquetas de entrenamiento marcadas con Make Sense
│   │   │   ├── val/               # Etiquetas de validación marcadas por Make Sense
│   │   │   ├── train.cache        # Archivos de caché generados al entrenarlo
│   │   │   ├── train.cache.npy
│   │   │   ├── val.cache
│   │   │   ├── val.cache.npy
│   ├── yolov5/                    # Código de YOLOv5 (descargado desde GitHub)
│   │   ├── data/
│   │   │   ├── coco128.yaml
│   │   ├── venv/
│   │   │   ├── Scripts/
│   │   │   │   ├── python.exe
│   ├── train_yolo.py              # Script de entrenamiento
│── External Libraries/            # Librerías externas utilizadas en el entorno
│── Scratches and Consoles/        # Consolas y borradores en el entorno de desarrollo
```

---

## 2. Creación del Proyecto

1. **Crear el proyecto y configurar la estructura básica:**
   - Crear el directorio del proyecto:
     ```bash
     mkdir Algoritmo-de-vision-computaciona
     ```
   - Entrar al directorio:
     ```bash
     cd Algoritmo-de-vision-computaciona
     ```

2. **Clonar el repositorio de YOLOv5:**
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   ```

---

## 3. Configuración del Entorno Virtual e Instalación de Paquetes

1. **Crear y activar el entorno virtual:**
   - Entrar al directorio de YOLOv5:
     ```bash
     cd yolov5
     ```
   - Crear el entorno virtual:
     ```bash
     python -m venv venv
     ```
   - Activar el entorno virtual:
     - En Windows:
       ```bash
       venv\Scripts\activate
       ```

2. **Instalar los paquetes necesarios:**
   - Instalar las dependencias listadas:
     ```bash
     pip install -r requirements.txt
     pip install roboflow
     ```

3. **Si se presenta algún error, instalar versiones compatibles:**
   ```bash
   pip install --upgrade protobuf==5.26.1
   pip install --upgrade pillow==10.2.0
   pip install --upgrade grpcio-tools==1.71.0
   pip install --upgrade moviepy==2.1.1
   ```
   - Luego volver a instalar las dependencias:
     ```bash
     pip install -r requirements.txt
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

4. **Probar la instalación de PyTorch:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

---

## 4. Preparación del Dataset

1. **Definir las clases o categorías:**
   - Ejemplo: Lentes, Lapicero y Celular.

2. **Descargar imágenes para cada clase:**
   - Utiliza la herramienta **'Download All Images'** (descargar extensión de Google):
     - Buscar la herramienta en la web.
     - Buscar la clase en la web (por ejemplo, "Lentes", "Lapicero", "Celular").
     - En la categoría de imágenes, cargar las imágenes que se quieran descargar (arrastrar el mouse para seleccionar muchas imágenes).
     - Activar la extensión para descargar todas las imágenes seleccionadas.
     - Guardar las imágenes en una carpeta temporal dentro del proyecto con el nombre de la clase.
     - Eliminar imágenes que no muestren correctamente el objeto o tengan formatos no deseados (se puede agrupar por tipo para facilitar la eliminación).

3. **Crear la estructura de carpetas para el dataset:**
   - Desde la raíz del proyecto:
     ```bash
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
     ```

4. **Dividir el dataset en 70-30:**
   - Colocar aproximadamente el 70% de las imágenes en `images/train` y el 30% en `images/val`.
   - Este proceso se repite para cada clase, es decir, todas las imágenes de todas las clases se colocan en las carpetas indicadas, pero en proporciones diferentes para entrenamiento y validación.

---

## 5. Proceso de Etiquetado con Make Sense

1. **Iniciar el etiquetado:**
   - Buscar la herramienta **'Make Sense'** en la web y acceder al primer enlace.
   - Dar click en **"Get Started"**.
   - Arrastrar la carpeta `images/train` al recuadro que aparece en la web (cuadro con el mensaje "Drop images").

2. **Configurar el etiquetado:**
   - Seleccionar la opción **'Object Detection'**.
   - En el recuadro de creación de etiquetas:
     - Dar click en el signo de **'+'** para agregar una nueva etiqueta.
     - Escribir el nombre del label y seleccionar el color.
     - Repetir el proceso para todas las clases o importar un archivo de texto con los labels.

3. **Etiquetar las imágenes:**
   - Iniciar el proyecto haciendo click en **"Start project"**.
   - Seleccionar el área exacta donde se encuentra el objeto en cada imagen.
   - Una vez etiquetado un objeto, el label se queda predefinido para las siguientes imágenes. Se recomienda etiquetar varias imágenes de una misma clase antes de pasar a la siguiente.

4. **Exportar las anotaciones:**
   - Al terminar de etiquetar, seleccionar la opción **'Actions'** y luego **'Export Annotations'**.
   - Seleccionar el formato **.zip de YOLO**.
   - Descomprimir el archivo y mover los archivos de extensión **.txt** a:
     - `labels/train` para las imágenes de entrenamiento.
     - Repetir el proceso con `images/val` y colocar los archivos en `labels/val`.

---

## 6. Creación del Archivo YAML de Configuración

1. **Crear el archivo YAML personalizado:**
   - Ubicación sugerida: `yolov5/data/scripts/`
   - Crear el archivo (por ejemplo, `custom.yaml`):
     ```bash
     cd Algoritmo-de-vision-computaciona/yolov5/data/scripts
     nano custom.yaml
     ```

2. **Contenido del archivo YAML (modificar los paths según tu máquina):**
   ```yaml
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
   ```

---

## 7. Configuración del Intérprete en el IDE (PyCharm)

1. **Seleccionar el intérprete de Python:**
   - Abrir la paleta de comandos con `Ctrl+Shift+P`.
   - Escribir **"Python interpreter"** y seleccionar la opción.
   - Agregar un nuevo intérprete mediante la ruta absoluta a `yolov5/venv/Scripts/python.exe`.
   - Si se permite la selección mediante el explorador de archivos, localizar y seleccionar el archivo `python.exe`.

---

## 8. Creación del Script de Entrenamiento (train_yolo.py)

1. **Crear el archivo en la raíz del proyecto (`Algoritmo-de-vision-computaciona`):**
   - Moverse a la carpeta adecuada o retroceder de carpeta:
     ```bash
     cd ..
     nano train_yolo.py
     ```
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

2. **Contenido del script (modificar los paths según tu máquina):**
   ```python
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
   ```

3. **Ejecutar el archivo en el IDE para iniciar el entrenamiento.**
   - Al finalizar el entrenamiento, el modelo se guardará en `yolov5/runs/train`.
   - Dentro de esta carpeta se generará una carpeta con las iniciales `expX` (donde X es un número creciente).
   - Dentro de `expX` se encuentran los archivos generados durante el entrenamiento, siendo el archivo `best.pt` el modelo entrenado.

---

## 9. Proceso de Inferencia o Prueba

1. **Crear un nuevo proyecto en PyCharm con las variables de entorno necesarias:**
   - Configurar el intérprete a Python 3.8 (o el que corresponda).

2. **Crear la carpeta para el modelo:**
   ```bash
   mkdir model
   cd model
   ```
   - Copiar el modelo entrenado `best.pt` y el archivo `custom.yaml` en esta carpeta.

3. **Crear el archivo de inferencia (inference.py):**
   - Desde la raíz del proyecto:
     ```bash
     cd ..
     nano inference.py
     ```

4. **Contenido del script de inferencia (modificar los paths según tu máquina):**
   ```python
   # Importar librerías
   import torch
   import cv2
   import numpy as np
   import pandas

   # Leer el modelo
   model = torch.hub.load('ultralytics/yolov5', 'custom',
                          path='C:/carlos_enrique/video/model/cel-lapi-lent.pt')

   # Realizar la captura de video
   cap = cv2.VideoCapture(0)

   # Empezar la detección
   while True:
       ret, frame = cap.read()  # Leer cada frame de la captura

       # Realizar las detecciones
       detect = model(frame)

       # Mostrar la información de la detección
       info = detect.pandas().xyxy[0]
       print(info)

       # Mostrar el resultado con FPS
       cv2.imshow('Detector de lentes, celular y lapicero', np.squeeze(detect.render()))

       # Leer el teclado para salir (presionar Esc)
       t = cv2.waitKey(5)
       if t == 27:
           break

   cap.release()
   cv2.destroyAllWindows()
   ```

5. **Ejecutar el script de inferencia para verificar el funcionamiento del modelo.**

---

## 10. Comandos y Notas Adicionales

### Comandos Extra (útiles en caso de necesitar reiniciar procesos de Python)
```bash
taskkill /F /IM python.exe /T
taskkill /F /IM pythonw.exe /T
taskkill /F /IM python.exe /IM pythonw.exe
wmic process where "name='python.exe'" delete
wmic process where "name='pythonw.exe'" delete
Stop-Process -Name python -Force
Stop-Process -Name pythonw -Force
```

### Otros comandos de instalación y verificación:
```bash
pip install --upgrade protobuf==5.26.1
pip install --upgrade pillow==10.2.0
pip install --upgrade grpcio-tools==1.71.0
pip install --upgrade moviepy==2.1.1
pip install idna==3.7

# Para comprobar si Roboflow sigue necesitando idna==3.7:
pip show roboflow

pip install -r requirements.txt
pip install --force-reinstall -r requirements.txt

pip list | findstr torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Notas sobre el intérprete en el IDE:
- Si mueves alguna carpeta o el intérprete cambia, actualiza la ruta en la configuración del IDE:
  - Usa `Ctrl+Shift+P` y escribe **"Select Interpreter"** para cambiar o agregar el intérprete.
  - Selecciona el intérprete de Python (por ejemplo, la versión 3.8 o crea uno nuevo) utilizando la ruta al `python.exe` dentro del `venv/Scripts` de tu proyecto.

---

Este README reorganizado conserva todo el contenido original y presenta cada paso de forma secuencial y clara, facilitando la configuración del proyecto, el entrenamiento del modelo y el proceso de inferencia.