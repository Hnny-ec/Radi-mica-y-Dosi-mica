from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# 0. CONFIGURACION DE SALIDA (PDF)

ruta_pdf_kmeans = r"C:\Users\lesse\Downloads\BONGO\segmentacion_kmeans.pdf"

pdf_kmeans = PdfPages(ruta_pdf_kmeans)


def guardar_figura(fig):
    """Guarda una figura como página del PDF y libera memoria."""
    pdf_kmeans.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# 1. CARGAR LA IMAGEN Y EXTRAER LA ROI

imagen = Image.open(
    r"C:\Users\lesse\Downloads\BONGO\Imagen.jpg"
)

imagen_array = np.array(imagen)

x1, y1 = 138, 117
x2, y2 = 314, 367

roi = imagen_array[y1:y2, x1:x2]

roi_gris = np.mean(
    roi,
    axis=2
).astype(np.uint8)

print("Tamaño de la ROI en escala de grises:", roi_gris.shape)


# 2. VECTOR DE CARACTERISTICAS v
# v: vector con M*N*P*d componentes.
#  d = 1 (solo se usa la intensidad de gris como característica) y P = 1 (imagen 2D, no un volumen).
# Cada componente v_i es entonces simplemente la intensidad de un pixel de la ROI.

M, N = roi_gris.shape
P = 1
d = 1

v = roi_gris.reshape(-1, 1).astype(float)

N_p = v.shape[0]  # M * N * P

print(f"M={M}, N={N}, P={P}, d={d}  ->  M*N*P*d = {N_p * d}")

# 3. ALGORITMO K-MEANS

def kmeans_manual(v, centroides_iniciales, max_iter=100, tol=1e-6):

    centroides = centroides_iniciales.astype(float).copy()

    K = centroides.shape[0]

    historial_Q = []

    for iteracion in range(max_iter):

        distancias = (v - centroides.reshape(1, K)) ** 2

        etiquetas = np.argmin(distancias, axis=1)

        # Calcular Q

        Q = 0.0

        for j in range(K):

            pixeles_clase = v[etiquetas == j]

            if pixeles_clase.shape[0] > 0:

                Q += np.sum(
                    (pixeles_clase - centroides[j]) ** 2
                )

        historial_Q.append(Q)

        nuevos_centroides = centroides.copy()

        for j in range(K):

            pixeles_clase = v[etiquetas == j]

            if pixeles_clase.shape[0] > 0:

                nuevos_centroides[j] = pixeles_clase.mean()

        cambio = np.max(
            np.abs(nuevos_centroides - centroides)
        )

        centroides = nuevos_centroides

        if cambio < tol:
            break

    return etiquetas, centroides.ravel(), historial_Q, iteracion + 1


# 4. VARIANZA INTRA-CLASE Y ENTRE-CLASES

def calcular_varianzas(v, etiquetas):
    """
    - varianza intra-clase de cada clase (mínima deseada)
    - varianza intra-clase promedio ponderada (pooled)
    - varianza entre-clases (máxima deseada, criterio de Fisher)
    """

    datos = v.ravel()

    media_global = datos.mean()

    N_total = datos.shape[0]

    varianzas_intra = {}

    var_intra_ponderada = 0.0

    var_entre = 0.0

    for clase in np.unique(etiquetas):

        pixeles_clase = datos[etiquetas == clase]

        n_clase = pixeles_clase.shape[0]

        media_clase = pixeles_clase.mean()

        var_clase = pixeles_clase.var()

        varianzas_intra[clase] = var_clase

        var_intra_ponderada += (n_clase / N_total) * var_clase

        var_entre += (
            (n_clase / N_total) * (media_clase - media_global) ** 2
        )

    return varianzas_intra, var_intra_ponderada, var_entre

# 5. CONDICIONES INICIALES

K = 2

condiciones_iniciales = [
    (
        "Separados bajos (10 y 40)",
        np.array([10.0, 40.0])
    ),
    (
        "Separados medios (20 y 80)",
        np.array([20.0, 80.0])
    ),
    (
        "Separados altos (60 y 100)",
        np.array([60.0, 100.0])
    ),
    (
        "Cercanos bajos (40 y 45)",
        np.array([40.0, 45.0])
    ),
    (
        "Cercanos medios (45 y 50)",
        np.array([45.0, 50.0])
    ),
    (
        "Invertidos (80 y 20)",
        np.array([80.0, 20.0])
    ),
]

# 6. EJECUTAR K-MEANS PARA CADA CONDICION INICIAL

resultados_kmeans = []

for nombre_condicion, centroides_iniciales in condiciones_iniciales:

    etiquetas, centroides, historial_Q, n_iter = kmeans_manual(
        v,
        centroides_iniciales
    )

    # Ordenar las clases de más oscura a más clara

    orden = np.argsort(centroides)

    mapa_orden = {
        clase_original: nuevo_indice
        for nuevo_indice, clase_original in enumerate(orden)
    }

    etiquetas_ordenadas = np.array(
        [mapa_orden[e] for e in etiquetas]
    )

    centroides_ordenados = centroides[orden]

    imagen_segmentada = etiquetas_ordenadas.reshape(
        roi_gris.shape
    )

    varianzas_intra, var_intra_ponderada, var_entre = (
        calcular_varianzas(v, etiquetas_ordenadas)
    )

    resultados_kmeans.append({
        "nombre": nombre_condicion,
        "centroides_iniciales": centroides_iniciales,
        "centroides_finales": centroides_ordenados,
        "etiquetas": etiquetas_ordenadas,
        "imagen_segmentada": imagen_segmentada,
        "Q_final": historial_Q[-1],
        "n_iteraciones": n_iter,
        "varianzas_intra": varianzas_intra,
        "var_intra_ponderada": var_intra_ponderada,
        "var_entre": var_entre
    })

# 7. TABLA COMPARATIVA

print("\n")
print("=" * 100)
print("SEGMENTACION K-MEANS (2 CLASES) - COMPARACION DE CONDICIONES INICIALES")
print("=" * 100)

print(
    f"{'Condición inicial':<34}"
    f"{'Centroides finales':<20}"
    f"{'Iteraciones':<13}"
    f"{'Q (WCSS total)':<17}"
    f"{'Var. intra':<13}"
    f"{'Var. entre':<12}"
)

print("-" * 100)

for r in resultados_kmeans:

    centroides_str = (
        f"[{r['centroides_finales'][0]:.1f}, "
        f"{r['centroides_finales'][1]:.1f}]"
    )

    print(
        f"{r['nombre']:<34}"
        f"{centroides_str:<20}"
        f"{r['n_iteraciones']:<13}"
        f"{r['Q_final']:<17.1f}"
        f"{r['var_intra_ponderada']:<13.4f}"
        f"{r['var_entre']:<12.4f}"
    )

print("-" * 100)


fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
ax.axis("off")

encabezados_km = [
    "Condición inicial",
    "Centroides iniciales",
    "Centroides finales",
    "Iteraciones",
    "Q (WCSS total)",
    "Var. intra (pond.)",
    "Var. entre"
]

filas_km = [
    [
        r["nombre"],
        f"[{r['centroides_iniciales'][0]:.0f}, {r['centroides_iniciales'][1]:.0f}]",
        f"[{r['centroides_finales'][0]:.1f}, {r['centroides_finales'][1]:.1f}]",
        str(r["n_iteraciones"]),
        f"{r['Q_final']:.1f}",
        f"{r['var_intra_ponderada']:.4f}",
        f"{r['var_entre']:.4f}"
    ]
    for r in resultados_kmeans
]

tabla_km = ax.table(
    cellText=filas_km,
    colLabels=encabezados_km,
    loc="center",
    cellLoc="center"
)

tabla_km.auto_set_font_size(False)
tabla_km.set_fontsize(8)
tabla_km.scale(1, 1.6)

ax.set_title(
    "Comparación de resultados k-means (2 clases) por condición inicial",
    pad=20
)

guardar_figura(fig)


# 8. PARA CADA CONDICION: IMAGEN SEGMENTADA E HISTOGRAMAS

for r in resultados_kmeans:

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 5),
        constrained_layout=True
    )

    axes[0].imshow(
        r["imagen_segmentada"],
        cmap="gray"
    )

    axes[0].set_title("Segmentación (2 clases)", fontsize=10)
    axes[0].set_xlabel("X (píxeles)", fontsize=8)
    axes[0].set_ylabel("Y (píxeles)", fontsize=8)

    for clase in [0, 1]:

        pixeles_clase = v.ravel()[
            r["etiquetas"] == clase
        ]

        N_Bi_clase, intensidades_clase = np.histogram(
            pixeles_clase,
            bins=256,
            range=(0, 256)
        )

        H_i_clase = N_Bi_clase / N_Bi_clase.sum()

        axes[clase + 1].bar(
            intensidades_clase[:-1],
            H_i_clase,
            width=0.8,
            align="center"
        )

        axes[clase + 1].set_title(
            f"Histograma clase {clase} "
            f"(centroide = {r['centroides_finales'][clase]:.1f})",
            fontsize=10
        )

        axes[clase + 1].set_xlabel("Intensidad de gris", fontsize=8)
        axes[clase + 1].set_ylabel("Frecuencia relativa", fontsize=8)
        axes[clase + 1].set_xlim(0, 150)
        axes[clase + 1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Condición inicial: {r['nombre']}",
        fontsize=13
    )

    guardar_figura(fig)


# 9. IDENTIFICAR LA MEJOR CONDICION INICIAL

# La mejor es la que da menor Q (WCSS total, equivalente amenor varianza intra-clase ponderada) y mayor varianza entre.

mejor = min(
    resultados_kmeans,
    key=lambda r: r["Q_final"]
)

print("\n")
print("=" * 60)
print("MEJOR CONDICION INICIAL (menor Q / menor varianza intra)")
print("=" * 60)
print("Condición:", mejor["nombre"])
print("Centroides finales:", mejor["centroides_finales"])
print("Q (WCSS total):", mejor["Q_final"])
print("Varianza intra ponderada:", mejor["var_intra_ponderada"])
print("Varianza entre clases:", mejor["var_entre"])
print("=" * 60)


pdf_kmeans.close()

print(f"\nPDF de segmentación k-means generado en: {ruta_pdf_kmeans}")