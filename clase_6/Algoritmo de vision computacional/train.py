import os
import subprocess

# Usa raw string para evitar problemas con secuencias de escape
dataset_path = r"C:\carlos_enrique\seminarios\clase_6\Algoritmo de vision computacional\data"  # Ajusta según la descarga

# Construye el comando de entrenamiento usando raw strings o barras dobles en las rutas
command = [
    "python", "train.py",
    "--img", "566",
    "--batch", "6",
    "--epochs", "70",
    "--data", os.path.join(dataset_path, r"yolov5\data\custom.yaml"),
    "--weights", "yolov5x6.pt",
    "--cache"
]

# Ejecuta el comando
subprocess.run(command)
