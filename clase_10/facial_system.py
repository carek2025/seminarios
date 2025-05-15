import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tkinter import Tk, Label, Button, Entry, messagebox, Frame
from PIL import Image, ImageTk
import threading

class FacialLoginCNNApp:
    def __init__(self):
        self.root = Tk()
        self.root.title("Sistema de Login Facial con CNN")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c3e50")

        self.dataset_dir = "dataset"
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        self.model_file = "facial_cnn_model.h5"
        self.model = None

        # Cargar el modelo DNN de OpenCV para detección de rostros
        try:
            self.face_net = cv2.dnn.readNetFromCaffe(
                r"C:\carlos_enrique\seminarios\clase_10\deploy.prototxt.txt",
                r"C:\carlos_enrique\seminarios\clase_10\res10_300x300_ssd_iter_140000.caffemodel"
            )
        except cv2.error as e:
            messagebox.showerror("Error", f"No se pudieron cargar los archivos del modelo DNN: {e}")
            exit()

        self.cap = None
        self.is_capturing = False

        # Interfaz principal
        self.main_frame = Frame(self.root, bg="#2c3e50")
        self.main_frame.pack(expand=True)

        self.label = Label(self.main_frame, text="Sistema de Login Facial", font=("Arial", 20), bg="#2c3e50", fg="white")
        self.label.pack(pady=20)

        self.create_button = Button(self.main_frame, text="Crear Cuenta", command=self.show_create_account, font=("Arial", 14), bg="#3498db", fg="white")
        self.create_button.pack(pady=10)

        self.login_button = Button(self.main_frame, text="Iniciar Sesión", command=self.show_login, font=("Arial", 14), bg="#3498db", fg="white")
        self.login_button.pack(pady=10)

        self.exit_button = Button(self.main_frame, text="Salir", command=self.exit_app, font=("Arial", 14), bg="#e74c3c", fg="white")
        self.exit_button.pack(pady=10)

        # Frame para creación de cuenta
        self.create_frame = Frame(self.root, bg="#2c3e50")
        self.create_label = Label(self.create_frame, text="Crear Cuenta", font=("Arial", 18), bg="#2c3e50", fg="white")
        self.create_label.pack(pady=10)

        self.name_entry_create = Entry(self.create_frame, font=("Arial", 14))
        self.name_entry_create.pack(pady=10)
        self.name_entry_create.insert(0, "Nombre de usuario")

        self.video_label_create = Label(self.create_frame, bg="#2c3e50")
        self.video_label_create.pack(pady=10)

        self.capture_button = Button(self.create_frame, text="Capturar Rostro", command=self.start_capture, font=("Arial", 12), bg="#3498db", fg="white")
        self.capture_button.pack(pady=5)

        self.train_button = Button(self.create_frame, text="Entrenar Modelo", command=self.train_model, font=("Arial", 12), bg="#3498db", fg="white")
        self.train_button.pack(pady=5)

        self.back_button_create = Button(self.create_frame, text="Volver", command=self.show_main, font=("Arial", 12), bg="#e74c3c", fg="white")
        self.back_button_create.pack(pady=5)

        # Frame para inicio de sesión
        self.login_frame = Frame(self.root, bg="#2c3e50")
        self.login_label = Label(self.login_frame, text="Iniciar Sesión", font=("Arial", 18), bg="#2c3e50", fg="white")
        self.login_label.pack(pady=10)

        self.name_entry_login = Entry(self.login_frame, font=("Arial", 14))
        self.name_entry_login.pack(pady=10)
        self.name_entry_login.insert(0, "Nombre de usuario")

        self.video_label_login = Label(self.login_frame, bg="#2c3e50")
        self.video_label_login.pack(pady=10)

        self.login_start_button = Button(self.login_frame, text="Iniciar Login", command=self.start_login, font=("Arial", 12), bg="#3498db", fg="white")
        self.login_start_button.pack(pady=5)

        self.back_button_login = Button(self.login_frame, text="Volver", command=self.show_main, font=("Arial", 12), bg="#e74c3c", fg="white")
        self.back_button_login.pack(pady=5)

        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_file):
            self.model = tf.keras.models.load_model(self.model_file)

    def show_main(self):
        self.create_frame.pack_forget()
        self.login_frame.pack_forget()
        self.main_frame.pack(expand=True)
        self.stop_capture()

    def show_create_account(self):
        self.main_frame.pack_forget()
        self.login_frame.pack_forget()
        self.create_frame.pack(expand=True)

    def show_login(self):
        self.main_frame.pack_forget()
        self.create_frame.pack_forget()
        self.login_frame.pack(expand=True)

    def exit_app(self):
        self.stop_capture()
        self.root.destroy()

    def detect_face(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                return (y, x2, y2, x)  # top, right, bottom, left
        return None

    def start_capture(self):
        name = self.name_entry_create.get().strip()
        if not name or name == "Nombre de usuario":
            messagebox.showerror("Error", "Por favor, ingresa un nombre válido.")
            return

        person_dir = os.path.join(self.dataset_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        elif len(os.listdir(person_dir)) > 0:
            messagebox.showerror("Error", "El usuario ya existe. Usa otro nombre.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo acceder a la cámara.")
            return

        self.is_capturing = True
        count = 0
        threading.Thread(target=self.capture_images, args=(name, person_dir, count)).start()
        self.update_video_create()

    def capture_images(self, name, person_dir, count):
        while self.is_capturing and count < 100:
            ret, frame = self.cap.read()
            if not ret:
                break
            face_location = self.detect_face(frame)
            if face_location:
                count += 1
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (224, 224))
                img_path = os.path.join(person_dir, f"{name}_{count}.jpg")
                cv2.imwrite(img_path, face_image)
                print(f"Imagen guardada: {img_path}")
            cv2.waitKey(100)
        self.stop_capture()
        if count >= 100:
            messagebox.showinfo("Éxito", "Imágenes capturadas. Ahora entrena el modelo.")

    def update_video_create(self):
        if self.is_capturing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (400, 300))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label_create.imgtk = imgtk
                self.video_label_create.configure(image=imgtk)
            self.root.after(10, self.update_video_create)

    def update_video_login(self):
        if self.is_capturing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (400, 300))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label_login.imgtk = imgtk
                self.video_label_login.configure(image=imgtk)
            self.root.after(10, self.update_video_login)

    def stop_capture(self):
        self.is_capturing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label_create.configure(image='')
        self.video_label_login.configure(image='')

    def train_model(self):
        name = self.name_entry_create.get().strip()
        if not name or name == "Nombre de usuario":
            messagebox.showerror("Error", "Por favor, ingresa un nombre válido.")
            return

        person_dir = os.path.join(self.dataset_dir, name)
        if not os.path.exists(person_dir) or not os.listdir(person_dir):
            messagebox.showerror("Error", "No hay imágenes para entrenar. Captura imágenes primero.")
            return

        # Verificar el dataset
        classes = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]
        if len(classes) < 2:
            messagebox.showerror("Error", "Necesitas al menos 2 usuarios con imágenes para entrenar.")
            return

        total_images = sum(len(os.listdir(os.path.join(self.dataset_dir, c))) for c in classes)
        print(f"Total de clases: {len(classes)}, Total de imágenes: {total_images}")

        # Configuración del generador de datos
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        try:
            train_generator = train_datagen.flow_from_directory(
                self.dataset_dir,
                target_size=(224, 224),
                batch_size=16,  # Reducido para datasets pequeños
                class_mode='categorical',
                subset='training',
                shuffle=True
            )

            validation_generator = train_datagen.flow_from_directory(
                self.dataset_dir,
                target_size=(224, 224),
                batch_size=16,
                class_mode='categorical',
                subset='validation',
                shuffle=True
            )

            print(f"Clases encontradas: {train_generator.class_indices}")
            print(f"Imágenes de entrenamiento: {train_generator.samples}")
            print(f"Imágenes de validación: {validation_generator.samples}")

            if train_generator.samples < 10 or validation_generator.samples < 2:
                messagebox.showerror("Error", "Dataset demasiado pequeño. Captura más imágenes por usuario.")
                return

            # Configurar el modelo
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            base_model.trainable = False

            inputs = Input(shape=(224, 224, 3))
            x = base_model(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)  # Añadir capa intermedia
            outputs = Dense(len(train_generator.class_indices), activation='softmax')(x)
            self.model = Model(inputs, outputs)

            # Compilar el modelo
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Entrenar el modelo
            history = self.model.fit(
                train_generator,
                epochs=50,  # Reducido para pruebas
                validation_data=validation_generator,
                verbose=1
            )

            # Guardar el modelo
            self.model.save(self.model_file)
            messagebox.showinfo("Éxito", "Modelo CNN entrenado y guardado.")
            print("Clases:", train_generator.class_indices)

            # Mostrar métricas finales
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            print(f"Precisión final de entrenamiento: {final_train_acc:.4f}")
            print(f"Precisión final de validación: {final_val_acc:.4f}")

        except Exception as e:
            messagebox.showerror("Error", f"Error durante el entrenamiento: {str(e)}")

    def start_login(self):
        name = self.name_entry_login.get().strip()
        if not name or name == "Nombre de usuario":
            messagebox.showerror("Error", "Por favor, ingresa un nombre válido.")
            return

        if not self.model:
            messagebox.showerror("Error", "No hay modelo entrenado. Crea una cuenta y entrena primero.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo acceder a la cámara.")
            return

        self.is_capturing = True
        threading.Thread(target=self.login_process, args=(name,)).start()
        self.update_video_login()

    def login_process(self, name):
        while self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                break
            face_location = self.detect_face(frame)
            if face_location:
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (224, 224))
                face_image = face_image.astype('float32') / 255.0
                face_image = np.expand_dims(face_image, axis=0)

                predictions = self.model.predict(face_image, verbose=0)
                predicted_class = np.argmax(predictions[0])
                # Mapear índice de clase a nombre
                train_datagen = ImageDataGenerator()
                train_generator = train_datagen.flow_from_directory(
                    self.dataset_dir,
                    target_size=(224, 224),
                    batch_size=32,
                    class_mode='categorical'
                )
                class_indices = train_generator.class_indices
                predicted_name = [k for k, v in class_indices.items() if v == predicted_class][0]

                if predicted_name == name:
                    messagebox.showinfo("Login Exitoso", f"Bienvenido, {name}!")
                    self.stop_capture()
                    self.show_main()
                    return
                else:
                    messagebox.showerror("Error", "Rostro no identificado.")
                    self.stop_capture()
                    self.show_main()
                    return
            cv2.waitKey(100)
        self.stop_capture()
        messagebox.showerror("Error", "Rostro no identificado.")
        self.show_main()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FacialLoginCNNApp()
    app.run()