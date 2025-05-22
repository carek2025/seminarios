import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from IPython.display import display
from io import StringIO

# Cargar el dataset Boston Housing
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

# Guardar los datos sin normalizar para mostrar ejemplos al usuario
X_test_raw = X_test.copy()  # Guardar versión sin normalizar para los ejemplos

# Nombres de las características
feature_names = [
    "CRIM (tasa de criminalidad per cápita)",
    "ZN (proporción de terreno residencial)",
    "INDUS (proporción de acres industriales)",
    "CHAS (1 si está cerca del río, 0 si no)",
    "NOX (concentración de óxidos de nitrógeno)",
    "RM (número promedio de habitaciones)",
    "AGE (proporción de casas antiguas)",
    "DIS (distancia a centros de empleo)",
    "RAD (índice de accesibilidad a autopistas)",
    "TAX (tasa de impuestos)",
    "PTRATIO (razón alumnos/profesor)",
    "B (proporción de población afroamericana)",
    "LSTAT (% de población de bajo estatus)"
]

# Convertir a DataFrame para visualización
df_train = pd.DataFrame(X_train, columns=feature_names)
df_train['PRICE'] = y_train
df_test = pd.DataFrame(X_test, columns=feature_names)
df_test['PRICE'] = y_test

# Mostrar el conjunto de entrenamiento como tabla interactiva
print("\nConjunto de entrenamiento completo (tabla interactiva):")
display(df_train)

# Mostrar el conjunto de prueba como tabla interactiva
print("\nConjunto de prueba completo (tabla interactiva):")
display(df_test)

# Mostrar como CSV (primeras 5 filas)
print("\nConjunto de entrenamiento (primeras 5 filas en formato CSV):\n")
with StringIO() as buffer:
    df_train.head().to_csv(buffer, index=False)
    print(buffer.getvalue())

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el modelo
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu', name='oculta1'),
    Dense(32, activation='relu', name='oculta2'),
    Dense(1, activation='linear', name='salida')
])

# Compilar y entrenar el modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

# Probar con una entrada del conjunto de prueba
test_input = X_test[0]
prediction = model.predict(np.array([test_input]))[0][0]
print(f"\nPrecio predicho (ejemplo de prueba): {prediction:.2f} (miles de dólares)")
print(f"Precio real: {y_test[0]:.2f} (miles de dólares)")

# Preguntar al usuario por los valores de las características
print("\nIngresa los valores para predecir el precio de una casa:")
user_input = []
for i, feature in enumerate(feature_names):
    example_value = X_test_raw[0][i]  # Valor sin normalizar de X_test[0]
    while True:
        try:
            value = float(input(f"{feature} (ejemplo: {example_value:.4f}): "))
            user_input.append(value)
            break
        except ValueError:
            print("Por favor, ingresa un número válido.")

# Normalizar la entrada del usuario
user_input = np.array([user_input])
user_input_normalized = scaler.transform(user_input)

# Hacer predicción con la entrada del usuario
user_prediction = model.predict(user_input_normalized)[0][0]
print(f"\nPrecio predicho para tu entrada: {user_prediction:.2f} (miles de dólares)")
