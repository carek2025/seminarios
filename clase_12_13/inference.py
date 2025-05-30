import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from ttkbootstrap import Style
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import datetime
import os
from tkinter import messagebox
import threading

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detector")
        self.style = Style(theme='flatly')  # Cambia 'darkly' por un tema disponible # Tema moderno
        
        # Inicializar variables
        self.modelo = load_model(r"C:\carlos_enrique\modelo-de-deteccion-de-emociones-python-tensorflow-dataset\modelo_emociones_optimizado.h5")
        self.emociones = ["Ira", "Asco", "Miedo", "Felicidad", "Tristeza", "Sorpresa", "Neutral"]
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.cap = None
        self.running = False
        self.camera_index = 0
        self.fullscreen = False
        
        # Directorio para guardar capturas
        self.screenshot_dir = "emotion_screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Crear interfaz
        self.setup_ui()
        
        # Iniciar actualización de video
        self.update_video()

    def setup_ui(self):
        # Frame principal
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Canvas para video
        self.canvas = tk.Canvas(self.main_frame, width=640, height=480)
        self.canvas.pack(side=TOP, fill=BOTH, expand=True)

        # Frame para controles
        self.controls_frame = tk.Frame(self.main_frame)
        self.controls_frame.pack(side=TOP, fill=X, pady=10)

        # Botones
        self.btn_start = tk.Button(self.controls_frame, text="Iniciar", style='primary.TButton', 
                                 command=self.start_capture)
        self.btn_start.pack(side=LEFT, padx=5)

        self.btn_stop = tk.Button(self.controls_frame, text="Pausar", style='secondary.TButton', 
                                command=self.stop_capture, state=DISABLED)
        self.btn_stop.pack(side=LEFT, padx=5)

        self.btn_screenshot = tk.Button(self.controls_frame, text="Captura", style='info.TButton', 
                                      command=self.take_screenshot, state=DISABLED)
        self.btn_screenshot.pack(side=LEFT, padx=5)

        self.btn_switch = tk.Button(self.controls_frame, text="Cambiar Cámara", style='warning.TButton',
                                  command=self.switch_camera)
        self.btn_switch.pack(side=LEFT, padx=5)

        self.btn_fullscreen = tk.Button(self.controls_frame, text="Pantalla Completa", style='success.TButton',
                                      command=self.toggle_fullscreen)
        self.btn_fullscreen.pack(side=LEFT, padx=5)

        # Frame para probabilidades
        self.prob_frame = tk.Frame(self.main_frame)
        self.prob_frame.pack(side=TOP, fill=X, pady=10)

        # Etiquetas para probabilidades
        self.prob_labels = {}
        for emotion in self.emociones:
            frame = tk.Frame(self.prob_frame)
            frame.pack(side=LEFT, padx=5)
            tk.Label(frame, text=f"{emotion}:", width=10).pack(side=LEFT)
            self.prob_labels[emotion] = tk.Label(frame, text="0%", width=10)
            self.prob_labels[emotion].pack(side=LEFT)

        # Área de registro
        self.log_text = tk.Text(self.main_frame, height=5, state='disabled')
        self.log_text.pack(side=TOP, fill=X, pady=10)

    def start_capture(self):
        if not self.running:
            try:
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "No se pudo acceder a la cámara")
                    return
                self.running = True
                self.btn_start.config(state=DISABLED)
                self.btn_stop.config(state=NORMAL)
                self.btn_screenshot.config(state=NORMAL)
                self.log("Captura iniciada")
            except Exception as e:
                messagebox.showerror("Error", f"Error al iniciar captura: {str(e)}")

    def stop_capture(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
            self.btn_start.config(state=NORMAL)
            self.btn_stop.config(state=DISABLED)
            self.btn_screenshot.config(state=DISABLED)
            self.log("Captura pausada")

    def switch_camera(self):
        self.camera_index = (self.camera_index + 1) % 2  # Alternar entre cámaras 0 y 1
        if self.running:
            self.stop_capture()
            self.start_capture()
        self.log(f"Cámara cambiada a índice {self.camera_index}")

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes('-fullscreen', self.fullscreen)
        self.log("Modo pantalla completa " + ("activado" if self.fullscreen else "desactivado"))

    def take_screenshot(self):
        if hasattr(self, 'current_frame'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.screenshot_dir, f"emotion_{timestamp}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
            self.log(f"Captura guardada: {filename}")

    def log(self, message):
        self.log_text.config(state='normal')
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def update_video(self):
        if self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rostros = self.face_cascade.detectMultiScale(gris, 1.3, 5)

                for (x, y, w, h) in rostros:
                    rostro = gris[y:y + h, x:x + w]
                    rostro = cv2.resize(rostro, (48, 48))
                    rostro = rostro / 255.0
                    rostro = np.expand_dims(rostro, axis=(0, -1))

                    prediccion = self.modelo.predict(rostro)[0]
                    emocion = self.emociones[np.argmax(prediccion)]

                    # Actualizar probabilidades
                    for i, emotion in enumerate(self.emociones):
                        self.prob_labels[emotion].config(text=f"{prediccion[i]*100:.1f}%")

                    # Dibujar rectángulo y etiqueta
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    self.log(f"Emoción detectada: {emocion}")

                # Convertir frame para Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = ImageTk.PhotoImage(img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.canvas.img = img

        self.root.after(10, self.update_video)

    def run(self):
        self.root.mainloop()

    def __del__(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    app.run()