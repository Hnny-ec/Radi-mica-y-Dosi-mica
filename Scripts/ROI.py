from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar la imagen
imagen = Image.open(r"C:\Users\lesse\Downloads\BONGO\Imagen.jpg")

print("Formato:", imagen.format)
print("Modo:", imagen.mode)
print("Tamaño:", imagen.size)

imagen_array = np.array(imagen)

# 2. Coordenadas de la ROI
x1, y1 = 138, 117
x2, y2 = 314, 367

# 3. Extraer la ROI
roi = imagen_array[y1:y2, x1:x2]

print("Tamaño de la ROI:", roi.shape)

# 4. Mostrar la ROI
plt.figure(figsize=(6, 6))
plt.imshow(roi)
plt.xlabel("X (píxeles)")
plt.ylabel("Y (píxeles)")
plt.title("Región de interés (ROI)")
plt.show()