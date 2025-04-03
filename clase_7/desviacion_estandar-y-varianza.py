import numpy as np
import matplotlib.pyplot as plt

datos = [25, 30, 35, 40, 45, 50,60,70,54,53,14,43,45,46,47,43,54,65]

media = np.mean(datos)
mediana = np.median(datos)
varianza = np.var(datos)  # Varianza
desviacion_estandar = np.std(datos)  # Desviación estándar
print(f"Media: {media}")
print(f"Mediana: {mediana}")

plt.bar(range(len(datos)), datos, color='skyblue', label='Datos')

plt.axhline(media, color='red', linestyle='dashed', linewidth=5, label=f'Media: {media}')
plt.axhline(mediana, color='blue', linestyle='dashed', linewidth=2, label=f'Mediana: {mediana}')
plt.axhline(media + desviacion_estandar, color='green', label=f'Media + 1 Desviación Estándar: {media + desviacion_estandar:.2f}', linestyle='dotted', linewidth=2)
plt.axhline(media - desviacion_estandar, color='green', label=f'Media - 1 Desviación Estándar: {media - desviacion_estandar:.2f}', linestyle='dotted', linewidth=2)

plt.xlabel('Índices')
plt.ylabel('Valores')
plt.title('Gráfico de Barras')

plt.legend()
plt.show()

print(f"Media: {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Varianza: {varianza:.2f}")
print(f"Desviación Estándar: {desviacion_estandar:.2f}")
