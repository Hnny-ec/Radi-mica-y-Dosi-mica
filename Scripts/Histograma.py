from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. CARGAR LA IMAGEN

imagen = Image.open(
    r"C:\Users\lesse\Downloads\BONGO\Imagen.jpg"
)

print("Formato:", imagen.format)
print("Modo:", imagen.mode)
print("Tamaño:", imagen.size)

# 2. CONVERTIR LA IMAGEN A UNA MATRIZ

imagen_array = np.array(imagen)

# 3. DEFINIR LAS COORDENADAS DE LA ROI

x1, y1 = 138, 117
x2, y2 = 314, 367

# 4. EXTRAER LA ROI

roi = imagen_array[y1:y2, x1:x2]

print("Tamaño de la ROI:", roi.shape)

# 5. CONVERTIR LA ROI A ESCALA DE GRISES

roi_gris = np.mean(roi, axis=2).astype(np.uint8)

print("Tamaño de la ROI en escala de grises:", roi_gris.shape)
print("Intensidad mínima:", roi_gris.min())
print("Intensidad máxima:", roi_gris.max())

plt.figure(figsize=(6, 6))

plt.imshow(roi_gris, cmap="gray")

plt.xlabel("X (píxeles)")
plt.ylabel("Y (píxeles)")
plt.title("Región de interés (ROI)")

plt.show()

# 6. CALCULAR EL HISTOGRAMA

histograma, intensidades = np.histogram(
    roi_gris,
    bins=256,
    range=(0, 256)
)

# 7. NORMALIZAR EL HISTOGRAMA

histograma_normalizado = histograma / histograma.sum()

# 8. CALCULAR LA ENTROPÍA

H_i = histograma_normalizado[histograma_normalizado > 0]

S = -np.sum(H_i * np.log2(H_i))

print("Entropía de la ROI:", S, "bits")

# 9. GRAFICAR EL HISTOGRAMA NORMALIZADO

plt.figure(figsize=(12, 7))

plt.bar(
    intensidades[:-1],
    histograma_normalizado,
    width=0.8,
    align="center"
)

plt.xlabel("Intensidad de gris")
plt.ylabel("Frecuencia relativa")
plt.title("Histograma normalizado de la ROI")

plt.xlim(25, 108)

plt.grid(axis="y", alpha=0.3)

plt.show()