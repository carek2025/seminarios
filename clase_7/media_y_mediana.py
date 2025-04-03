import numpy as np
import matplotlib.pyplot as plt

datos = [25, 30, 35, 40, 45, 50]

media = np.mean(datos)
mediana = np.median(datos)
print(f"Media: {media}")
print(f"Mediana: {mediana}")

plt.bar(range(len(datos)), datos, color='skyblue', label='Datos')

plt.axhline(media, color='red', linestyle='dashed', linewidth=5, label=f'Media: {media}')
plt.axhline(mediana, color='blue', linestyle='dashed', linewidth=2, label=f'Mediana: {mediana}')


plt.xlabel('Índices')
plt.ylabel('Valores')
plt.title('Gráfico de Barras')

plt.legend()
plt.show()
