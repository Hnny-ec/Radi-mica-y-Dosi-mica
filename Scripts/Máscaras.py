from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.ndimage import gaussian_filter

# 0. CONFIGURACION DE SALIDA (PDF)

ruta_pdf = r"C:\Users\lesse\Downloads\BONGO\resultados_entropia.pdf"

# Segundo PDF: cómo varía la ROI (imagen original recortada)

ruta_pdf_imagen_completa = r"C:\Users\lesse\Downloads\BONGO\variacion_roi.pdf"

pdf = PdfPages(ruta_pdf)
pdf_imagen_completa = PdfPages(ruta_pdf_imagen_completa)


def guardar_figura(fig, documento=None):
    """Guarda una figura como página de un PDF y libera memoria."""
    if documento is None:
        documento = pdf
    documento.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# 1. CARGAR LA IMAGEN

imagen = Image.open(
    r"C:\Users\lesse\Downloads\BONGO\Imagen.jpg"
)

print("Formato:", imagen.format)
print("Modo:", imagen.mode)
print("Tamaño:", imagen.size)

imagen_array = np.array(imagen)

# 2. DEFINIR LA REGION DE INTERES (ROI)

x1, y1 = 138, 117
x2, y2 = 314, 367

roi = imagen_array[y1:y2, x1:x2]

print("Tamaño de la ROI:", roi.shape)


roi_gris = np.mean(
    roi,
    axis=2
).astype(np.uint8)

print(
    "Tamaño de la ROI en escala de grises:",
    roi_gris.shape
)

print(
    "Intensidad mínima:",
    roi_gris.min()
)

print(
    "Intensidad máxima:",
    roi_gris.max()
)

fig = plt.figure(figsize=(6, 6))

plt.imshow(
    roi_gris,
    cmap="gray"
)

plt.xlabel("X (píxeles)")
plt.ylabel("Y (píxeles)")

plt.title(
    "Región de interés (ROI)"
)

guardar_figura(fig)

N_Bi, intensidades = np.histogram(
    roi_gris,
    bins=256,
    range=(0, 256)
)

N_T = N_Bi.sum()

H_i = N_Bi / N_T

print(
    "Número total de píxeles:",
    N_T
)

print(
    "Suma del histograma normalizado:",
    H_i.sum()
)

# 3. ENTROPIA DE LA ROI ORIGINAL

H_no_cero = H_i[H_i > 0]

S_original = -np.sum(
    H_no_cero * np.log2(H_no_cero)
)

print(
    "Entropía de la ROI original:",
    S_original,
    "bits"
)


# 4. MOSTRAR EL HISTOGRAMA ORIGINAL

fig = plt.figure(figsize=(12, 7))

plt.bar(
    intensidades[:-1],
    H_i,
    width=0.8,
    align="center"
)

plt.xlabel("Intensidad de gris")

plt.ylabel("Frecuencia relativa")

plt.title(
    "Histograma normalizado de la ROI original"
)

plt.xlim(25, 108)

plt.grid(
    axis="y",
    alpha=0.3
)

guardar_figura(fig)

# 5. DEFINIR MASCARAS Y DESVIACIONES ESTANDAR

# Radio de la máscara:

# radio = 1  -> 3 x 3
# radio = 2  -> 5 x 5
# radio = 3  -> 7 x 7

radios = [1, 2, 3]

# Diferentes desviaciones estándar.

sigmas = [
    0.5,
    2.0,
    4.0
]


# 6. APLICAR TODAS LAS COMBINACIONES

resultados = []

imagenes_filtradas = {}

histogramas_filtrados = {}

entropias = {}


for radio in radios:

    for sigma in sigmas:

        # Filtro gaussiano

        roi_filtrada = gaussian_filter(
            roi_gris.astype(float),
            sigma=sigma,
            radius=radio
        )


        # Mantener valores entre 0 y 255

        roi_filtrada = np.clip(
            roi_filtrada,
            0,
            255
        ).astype(np.uint8)

        tamano_mascara = 2 * radio + 1

        N_Bi_filtrada, intensidades_filtrada = np.histogram(
            roi_filtrada,
            bins=256,
            range=(0, 256)
        )


        N_T_filtrada = N_Bi_filtrada.sum()


        H_i_filtrada = (
            N_Bi_filtrada / N_T_filtrada
        )

        # Entropía

        H_no_cero_filtrada = (
            H_i_filtrada[
                H_i_filtrada > 0
            ]
        )


        S_filtrada = -np.sum(
            H_no_cero_filtrada
            * np.log2(H_no_cero_filtrada)
        )

        # Diferencia de entropía

        diferencia_entropia = (
            S_filtrada - S_original
        )

        resultados.append([
            tamano_mascara,
            sigma,
            S_filtrada,
            diferencia_entropia
        ])

        clave = (
            tamano_mascara,
            sigma
        )

        imagenes_filtradas[clave] = (
            roi_filtrada
        )

        histogramas_filtrados[clave] = (
            intensidades_filtrada,
            H_i_filtrada
        )

        entropias[clave] = S_filtrada

# 7. TABLA DE RESULTADOS (consola)

print("\n")
print("=" * 70)
print("RESULTADOS DEL FILTRO GAUSSIANO")
print("=" * 70)

print(
    f"{'Máscara':<12}"
    f"{'Sigma':<12}"
    f"{'Entropía':<20}"
    f"{'Δ Entropía':<20}"
)

print("-" * 70)


for resultado in resultados:

    mascara = resultado[0]
    sigma = resultado[1]
    entropia = resultado[2]
    diferencia = resultado[3]

    print(
        f"{mascara}x{mascara:<9}"
        f"{sigma:<12.1f}"
        f"{entropia:<20.6f}"
        f"{diferencia:<20.6f}"
    )


print("-" * 70)

print(
    "Entropía original:",
    f"{S_original:.6f}",
    "bits"
)


fig, ax = plt.subplots(figsize=(9, 4))
ax.axis("off")

encabezados = ["Máscara", "Sigma", "Entropía (bits)", "Δ Entropía (bits)"]

filas = [
    [f"{r[0]}x{r[0]}", f"{r[1]:.1f}", f"{r[2]:.6f}", f"{r[3]:.6f}"]
    for r in resultados
]

tabla = ax.table(
    cellText=filas,
    colLabels=encabezados,
    loc="center",
    cellLoc="center"
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 1.5)

ax.set_title(
    f"Resultados del filtro gaussiano\nEntropía original: {S_original:.6f} bits",
    pad=20
)

guardar_figura(fig)


# 8.  IMAGENES FILTRADAS

fig, axes = plt.subplots(
    len(radios),
    len(sigmas),
    figsize=(15, 16),
    constrained_layout=True
)

for i, radio in enumerate(radios):

    for j, sigma in enumerate(sigmas):

        tamano_mascara = 2 * radio + 1

        imagen_mostrada = imagenes_filtradas[
            (tamano_mascara, sigma)
        ]

        axes[i, j].imshow(
            imagen_mostrada,
            cmap="gray"
        )

        axes[i, j].set_title(
            f"Máscara {tamano_mascara}×{tamano_mascara}, σ = {sigma}",
            fontsize=10
        )

        axes[i, j].set_xlabel(
            "X (píxeles)",
            fontsize=8
        )

        axes[i, j].set_ylabel(
            "Y (píxeles)",
            fontsize=8
        )

        axes[i, j].tick_params(labelsize=7)

fig.suptitle(
    "ROI filtrada con diferentes máscaras y desviaciones estándar",
    fontsize=14
)

guardar_figura(fig)


fig, axes = plt.subplots(
    len(radios),
    len(sigmas),
    figsize=(16, 16),
    constrained_layout=True
)

for i, radio in enumerate(radios):

    for j, sigma in enumerate(sigmas):

        tamano_mascara = 2 * radio + 1

        imagen_filtrada = imagenes_filtradas[
            (tamano_mascara, sigma)
        ]

        diferencia_img = (
            roi_gris.astype(int) - imagen_filtrada.astype(int)
        )

        im = axes[i, j].imshow(
            diferencia_img,
            cmap="seismic",
            vmin=-20,
            vmax=20
        )

        axes[i, j].set_title(
            f"Diferencia | {tamano_mascara}×{tamano_mascara}, σ = {sigma}",
            fontsize=10
        )

        axes[i, j].set_xlabel("X (píxeles)", fontsize=8)
        axes[i, j].set_ylabel("Y (píxeles)", fontsize=8)
        axes[i, j].tick_params(labelsize=7)


fig.suptitle(
    "Diferencia entre ROI original y ROI filtrada\n(lo que el filtro suavizó)",
    fontsize=14
)

fig.colorbar(
    im,
    ax=axes,
    shrink=0.8,
    label="Diferencia de intensidad"
)

guardar_figura(fig)

# 9. MOSTRAR LOS HISTOGRAMAS

fig, axes = plt.subplots(
    len(radios),
    len(sigmas),
    figsize=(16, 16),
    constrained_layout=True
)

for i, radio in enumerate(radios):

    for j, sigma in enumerate(sigmas):

        tamano_mascara = 2 * radio + 1

        intensidades_filtrada, H_i_filtrada = (
            histogramas_filtrados[
                (tamano_mascara, sigma)
            ]
        )

        axes[i, j].bar(
            intensidades_filtrada[:-1],
            H_i_filtrada,
            width=0.8,
            align="center"
        )

        axes[i, j].set_title(
            f"Máscara {tamano_mascara}×{tamano_mascara}, σ = {sigma}",
            fontsize=10
        )

        axes[i, j].set_xlabel("Intensidad de gris", fontsize=8)
        axes[i, j].set_ylabel("Frecuencia relativa", fontsize=8)
        axes[i, j].tick_params(labelsize=7)

        axes[i, j].set_xlim(25, 108)

        axes[i, j].grid(axis="y", alpha=0.3)


fig.suptitle(
    "Histogramas de las ROI filtradas",
    fontsize=14
)

guardar_figura(fig)

# 10. IDENTIFICAR LA MAYOR VARIACION DE ENTROPIA

resultado_mayor = max(
    resultados,
    key=lambda x: abs(x[3])
)

print("\n")
print("=" * 70)
print("MAYOR VARIACION ABSOLUTA DE ENTROPIA")
print("=" * 70)

print("Máscara:", f"{resultado_mayor[0]} × {resultado_mayor[0]}")
print("Sigma:", resultado_mayor[1])
print("Entropía:", resultado_mayor[2], "bits")
print("Diferencia respecto a la original:", resultado_mayor[3], "bits")
print("=" * 70)


# VARIACION DE LA ROI (segundo PDF)

# ROI original

fig = plt.figure(figsize=(6, 6), constrained_layout=True)

plt.imshow(
    roi_gris,
    cmap="gray"
)

plt.xlabel("X (píxeles)")
plt.ylabel("Y (píxeles)")

plt.title("ROI original (escala de grises)")

guardar_figura(fig, documento=pdf_imagen_completa)

fig, axes = plt.subplots(
    len(radios),
    len(sigmas),
    figsize=(15, 16),
    constrained_layout=True
)

for i, radio in enumerate(radios):

    for j, sigma in enumerate(sigmas):

        tamano_mascara = 2 * radio + 1

        imagen_mostrada = imagenes_filtradas[
            (tamano_mascara, sigma)
        ]

        axes[i, j].imshow(
            imagen_mostrada,
            cmap="gray"
        )

        axes[i, j].set_title(
            f"Máscara {tamano_mascara}×{tamano_mascara}, σ = {sigma}",
            fontsize=10
        )

        axes[i, j].set_xlabel("X (píxeles)", fontsize=8)
        axes[i, j].set_ylabel("Y (píxeles)", fontsize=8)
        axes[i, j].tick_params(labelsize=7)


fig.suptitle(
    "ROI filtrada con diferentes máscaras y desviaciones estándar",
    fontsize=14
)

guardar_figura(fig, documento=pdf_imagen_completa)



# Imágenes diferencia de la ROI

fig, axes = plt.subplots(
    len(radios),
    len(sigmas),
    figsize=(16, 16),
    constrained_layout=True
)

for i, radio in enumerate(radios):

    for j, sigma in enumerate(sigmas):

        tamano_mascara = 2 * radio + 1

        imagen_filtrada = imagenes_filtradas[
            (tamano_mascara, sigma)
        ]

        diferencia_img = (
            roi_gris.astype(int) - imagen_filtrada.astype(int)
        )

        im = axes[i, j].imshow(
            diferencia_img,
            cmap="seismic",
            vmin=-20,
            vmax=20
        )

        axes[i, j].set_title(
            f"Diferencia | {tamano_mascara}×{tamano_mascara}, σ = {sigma}",
            fontsize=10
        )

        axes[i, j].set_xlabel("X (píxeles)", fontsize=8)
        axes[i, j].set_ylabel("Y (píxeles)", fontsize=8)
        axes[i, j].tick_params(labelsize=7)


fig.suptitle(
    "Diferencia entre ROI original y ROI filtrada\n(lo que el filtro suavizó)",
    fontsize=14
)

fig.colorbar(
    im,
    ax=axes,
    shrink=0.8,
    label="Diferencia de intensidad"
)

guardar_figura(fig, documento=pdf_imagen_completa)

pdf.close()
pdf_imagen_completa.close()

print(f"\nPDF de resultados generado en: {ruta_pdf}")
print(f"PDF de variación de la ROI generado en: {ruta_pdf_imagen_completa}")