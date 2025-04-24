import os
import subprocess

command = [
    "python", "C:\\carlos_enrique\\seminarios\\clase_6\\Algoritmo de vision computacional\\yolov5\\train.py",
    "--img", "566",
    "--batch", "6",
    "--epochs", "70",
    "--data", "C:\\carlos_enrique\\seminarios\\clase_6\\Algoritmo de vision computacional\\yolov5\\data\\custom.yaml",
    "--weights", "yolov5x.pt",
    "--cache"
]

# Ejecutar el proceso con stdout y stderr visibles
process = subprocess.Popen(
    command,
    cwd="C:\\carlos_enrique\\seminarios\\clase_6\\Algoritmo-de-vision-computacional\\yolov5",
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