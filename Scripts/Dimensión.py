from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Cargar imagen
imagen = Image.open(r"C:\Users\lesse\Downloads\BONGO\Imagen.jpg")

# Mostrar imagen
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(imagen)
ax.set_xlabel("X (píxeles)")
ax.set_ylabel("Y (píxeles)")
ax.set_title("Selecciona la región de interés (ROI)")

# Función que se ejecuta al seleccionar el rectángulo
def seleccionar_roi(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    print("ROI seleccionada:")
    print(f"x1 = {x1:.0f}")
    print(f"y1 = {y1:.0f}")
    print(f"x2 = {x2:.0f}")
    print(f"y2 = {y2:.0f}")

selector = RectangleSelector(
    ax,
    seleccionar_roi,
    useblit=True,
    button=[1],
    minspanx=5,
    minspany=5,
    spancoords="pixels",
    interactive=True
)

plt.show()