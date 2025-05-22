import tensorflow as tf
import sklearn as sk
import numpy as np
import pandas as pd
from IPython.display import display
from io import StringIO

(X_train,y_train),(X_test, y_test)=tf.keras.datasets.boston_housing.load_data()
scaler=sk.preprocessing.StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

feature_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", 
    "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

# Convertir a DataFrame para visualización
df_train = pd.DataFrame(X_train, columns=feature_names)
df_train['PRICE'] = y_train
df_test = pd.DataFrame(X_test, columns=feature_names)
df_test['PRICE'] = y_test

print("\nConjunto de entrenamiento completo (tabla interactiva):")
display(df_train)  
print("\nConjunto de prueba completo (tabla interactiva):")
display(df_test)

print("\nConjunto de entrenamiento (primeras 5 filas en formato CSV):\n")
with StringIO() as buffer:
    df_train.head().to_csv(buffer, index=False)
    print(buffer.getvalue())

model= tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64,activation='relu', name='oculta1'),
    tf.keras.layers.Dense(32,activation='relu',name='oculta2'),
    tf.keras.layers.Dense(1,activation='linear',name='salida')
])

model.compile(optimizer='adam',loss='mse')
model.fit(X_train,y_train,epochs=500,batch_size=32,verbose=0)

test_input=X_test[0]
prediction=model.predict(np.array([test_input]))[0][0]
print(f'Precio predicho: {prediction:2f}')
print(f"Precio real: {y_test[0]:.2f}")
