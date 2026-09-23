from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# 0. CONFIGURACION DE SALIDA

ruta_pdf_radiomica = r"C:\Users\lesse\Downloads\BONGO\radiomica_glcm.pdf"

pdf_radiomica = PdfPages(ruta_pdf_radiomica)

EPS = 1e-10


def guardar_figura(fig):
    pdf_radiomica.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# 1. CARGAR LA IMAGEN Y EXTRAER LA ROI

imagen = Image.open(
    r"C:\Users\lesse\Downloads\BONGO\Imagen.jpg"
)

imagen_array = np.array(imagen)

x1, y1 = 138, 117
x2, y2 = 314, 367

roi = imagen_array[y1:y2, x1:x2]

roi_gris = np.mean(roi, axis=2).astype(np.uint8)

print("Tamaño de la ROI en escala de grises:", roi_gris.shape)


# 2. SEGMENTACION K-MEANS (K=2)
# Como ya se demostró en la parte 2, el resultado final es idéntico sin importar la condición inicial, así que aquí se
# corre una sola vez con una condición inicial representativa.

def kmeans_manual(v, centroides_iniciales, max_iter=100, tol=1e-6):

    centroides = centroides_iniciales.astype(float).copy()
    K = centroides.shape[0]

    for iteracion in range(max_iter):

        distancias = (v - centroides.reshape(1, K)) ** 2
        etiquetas = np.argmin(distancias, axis=1)

        nuevos_centroides = centroides.copy()

        for j in range(K):
            pixeles_clase = v[etiquetas == j]
            if pixeles_clase.shape[0] > 0:
                nuevos_centroides[j] = pixeles_clase.mean()

        cambio = np.max(np.abs(nuevos_centroides - centroides))
        centroides = nuevos_centroides

        if cambio < tol:
            break

    return etiquetas, centroides.ravel()


v = roi_gris.reshape(-1, 1).astype(float)

etiquetas, centroides = kmeans_manual(
    v,
    np.array([20.0, 80.0])
)

# Ordenar clases de más oscura a más clara (clase 0 = lóbulos, clase 1 = fondo)

orden = np.argsort(centroides)
mapa_orden = {c_orig: c_nuevo for c_nuevo, c_orig in enumerate(orden)}
etiquetas = np.array([mapa_orden[e] for e in etiquetas])
centroides = centroides[orden]

mascara_clases = etiquetas.reshape(roi_gris.shape)

print("Centroides finales:", centroides)

# 3. CARACTERISTICAS DE PRIMER ORDEN

def primer_orden(pixeles):

    pixeles = pixeles.astype(float)

    N_p = pixeles.shape[0]

    X_bar = pixeles.mean()

    # Entropía, a partir del histograma normalizado

    N_Bi, _ = np.histogram(pixeles, bins=256, range=(0, 256))

    p_i = N_Bi / N_Bi.sum()

    entropia = -np.sum(p_i * np.log2(p_i + EPS))

    # Skewness

    m2 = np.sum((pixeles - X_bar) ** 2) / N_p
    m3 = np.sum((pixeles - X_bar) ** 3) / N_p

    skewness = m3 / (np.sqrt(m2) ** 3)

    # Curtosis

    m4 = np.sum((pixeles - X_bar) ** 4) / N_p

    curtosis = m4 / (m2 ** 2)

    return {
        "N_p": N_p,
        "media": X_bar,
        "entropia": entropia,
        "skewness": skewness,
        "curtosis": curtosis
    }

# 4. MATRIZ GLCM (angulo = 0, d = 1)

# Para cada clase, se construye la GLCM usando solo los pares de pixeles vecinos horizontales (columna, columna+1) donde
# AMBOS pixeles pertenecen a la máscara de esa clase. La matriz se simetriza (se suma su transpuesta) para que sea
# invariante a la dirección del recorrido y luego se normaliza para obtener p(i,j).
# Además, se cuantizan los niveles de gris a NIVELES_GLCM bins para obtener una matriz pequeña y fácil de leer  

NIVELES_GLCM = 16


def cuantizar_imagen(roi_gris, nivel_min, nivel_max, n_niveles):
    """Convierte cada pixel de roi_gris (0-255) a un bin entero
    entre 0 y n_niveles-1, según su posición dentro del rango
    [nivel_min, nivel_max]."""

    bordes = np.linspace(nivel_min, nivel_max + 1, n_niveles + 1)

    imagen_cuantizada = np.digitize(roi_gris, bordes) - 1

    imagen_cuantizada = np.clip(imagen_cuantizada, 0, n_niveles - 1)

    centros_bin = (bordes[:-1] + bordes[1:]) / 2

    return imagen_cuantizada, centros_bin


def calcular_glcm(roi_gris, mascara, clase, nivel_min, nivel_max):

    Ng = nivel_max - nivel_min + 1

    glcm = np.zeros((Ng, Ng), dtype=float)

    filas, columnas = roi_gris.shape

    for i in range(filas):
        for j in range(columnas - 1):

            if mascara[i, j] == clase and mascara[i, j + 1] == clase:

                nivel_centro = roi_gris[i, j] - nivel_min
                nivel_vecino = roi_gris[i, j + 1] - nivel_min

                glcm[nivel_centro, nivel_vecino] += 1

    glcm = glcm + glcm.T

    suma_total = glcm.sum()

    if suma_total > 0:
        glcm_normalizada = glcm / suma_total
    else:
        glcm_normalizada = glcm

    return glcm_normalizada


def calcular_glcm_cuantizada(imagen_cuantizada, mascara, clase, n_niveles):
    """Misma lógica que calcular_glcm, pero sobre la imagen ya
    cuantizada a n_niveles bins (matriz pequeña, fácil de leer)."""

    glcm = np.zeros((n_niveles, n_niveles), dtype=float)

    filas, columnas = imagen_cuantizada.shape

    for i in range(filas):
        for j in range(columnas - 1):

            if mascara[i, j] == clase and mascara[i, j + 1] == clase:

                bin_centro = imagen_cuantizada[i, j]
                bin_vecino = imagen_cuantizada[i, j + 1]

                glcm[bin_centro, bin_vecino] += 1

    glcm = glcm + glcm.T

    suma_total = glcm.sum()

    if suma_total > 0:
        glcm_normalizada = glcm / suma_total
    else:
        glcm_normalizada = glcm

    return glcm_normalizada


def metricas_glcm(glcm_normalizada, nivel_min):

    Ng = glcm_normalizada.shape[0]

    niveles = np.arange(nivel_min, nivel_min + Ng)

    i_idx, j_idx = np.meshgrid(niveles, niveles, indexing="ij")

    # Autocorrelación: 

    autocorrelacion = np.sum(
        glcm_normalizada * i_idx * j_idx
    )

    # Contraste: 

    contraste = np.sum(
        glcm_normalizada * (i_idx - j_idx) ** 2
    )

    # Energía conjunta: 

    energia_conjunta = np.sum(
        glcm_normalizada ** 2
    )

    # Entropía conjunta:

    entropia_conjunta = -np.sum(
        glcm_normalizada * np.log2(glcm_normalizada + EPS)
    )

    return {
        "autocorrelacion": autocorrelacion,
        "contraste": contraste,
        "energia_conjunta": energia_conjunta,
        "entropia_conjunta": entropia_conjunta
    }


# 5. CALCULAR TODO PARA LAS DOS CLASES (SEGMENTACIONES)


nivel_min = int(roi_gris.min())
nivel_max = int(roi_gris.max())

imagen_cuantizada, centros_bin = cuantizar_imagen(
    roi_gris, nivel_min, nivel_max, NIVELES_GLCM
)

resultados_radiomicos = {}

for clase in [0, 1]:

    pixeles_clase = roi_gris[mascara_clases == clase]

    stats_1er_orden = primer_orden(pixeles_clase)

    glcm_norm = calcular_glcm(
        roi_gris, mascara_clases, clase, nivel_min, nivel_max
    )

    stats_glcm = metricas_glcm(glcm_norm, nivel_min)

    glcm_cuantizada = calcular_glcm_cuantizada(
        imagen_cuantizada, mascara_clases, clase, NIVELES_GLCM
    )

    resultados_radiomicos[clase] = {
        "pixeles": pixeles_clase,
        "primer_orden": stats_1er_orden,
        "glcm": glcm_norm,
        "glcm_metricas": stats_glcm,
        "glcm_cuantizada": glcm_cuantizada
    }

# 6. RESULTADOS

print("\n")
print("=" * 80)
print("CARACTERISTICAS RADIOMICAS POR CLASE (SEGMENTACION K-MEANS, K=2)")
print("=" * 80)

for clase in [0, 1]:

    r = resultados_radiomicos[clase]
    po = r["primer_orden"]
    gm = r["glcm_metricas"]

    nombre_clase = "Clase 0 (lóbulos oscuros)" if clase == 0 else "Clase 1 (fondo claro)"

    print(f"\n--- {nombre_clase} ---")
    print(f"N pixeles:            {po['N_p']}")
    print(f"Media:                {po['media']:.4f}")
    print(f"Entropía (1er orden): {po['entropia']:.6f} bits")
    print(f"Skewness:             {po['skewness']:.6f}")
    print(f"Curtosis:             {po['curtosis']:.6f}")
    print(f"Autocorrelación GLCM: {gm['autocorrelacion']:.4f}")
    print(f"Contraste GLCM:       {gm['contraste']:.4f}")
    print(f"Energía conjunta:     {gm['energia_conjunta']:.6f}")
    print(f"Entropía conjunta:    {gm['entropia_conjunta']:.6f} bits")

print("\n" + "=" * 80)


fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
ax.axis("off")

encabezados = [
    "Característica",
    "Clase 0 (lóbulos)",
    "Clase 1 (fondo)"
]

filas = [
    ["N pixeles",
     f"{resultados_radiomicos[0]['primer_orden']['N_p']}",
     f"{resultados_radiomicos[1]['primer_orden']['N_p']}"],
    ["Media",
     f"{resultados_radiomicos[0]['primer_orden']['media']:.2f}",
     f"{resultados_radiomicos[1]['primer_orden']['media']:.2f}"],
    ["Entropía (1er orden)",
     f"{resultados_radiomicos[0]['primer_orden']['entropia']:.4f}",
     f"{resultados_radiomicos[1]['primer_orden']['entropia']:.4f}"],
    ["Skewness",
     f"{resultados_radiomicos[0]['primer_orden']['skewness']:.4f}",
     f"{resultados_radiomicos[1]['primer_orden']['skewness']:.4f}"],
    ["Curtosis",
     f"{resultados_radiomicos[0]['primer_orden']['curtosis']:.4f}",
     f"{resultados_radiomicos[1]['primer_orden']['curtosis']:.4f}"],
    ["Autocorrelación (GLCM)",
     f"{resultados_radiomicos[0]['glcm_metricas']['autocorrelacion']:.2f}",
     f"{resultados_radiomicos[1]['glcm_metricas']['autocorrelacion']:.2f}"],
    ["Contraste (GLCM)",
     f"{resultados_radiomicos[0]['glcm_metricas']['contraste']:.4f}",
     f"{resultados_radiomicos[1]['glcm_metricas']['contraste']:.4f}"],
    ["Energía conjunta (GLCM)",
     f"{resultados_radiomicos[0]['glcm_metricas']['energia_conjunta']:.6f}",
     f"{resultados_radiomicos[1]['glcm_metricas']['energia_conjunta']:.6f}"],
    ["Entropía conjunta (GLCM)",
     f"{resultados_radiomicos[0]['glcm_metricas']['entropia_conjunta']:.4f}",
     f"{resultados_radiomicos[1]['glcm_metricas']['entropia_conjunta']:.4f}"],
]

tabla = ax.table(
    cellText=filas,
    colLabels=encabezados,
    loc="center",
    cellLoc="center"
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1, 1.6)

ax.set_title(
    "Características radiómicas por clase (primer orden + GLCM)",
    pad=20
)

guardar_figura(fig)


# 7. VISUALIZACION: MASCARA + GLCM POR CLASE

for clase in [0, 1]:

    r = resultados_radiomicos[clase]

    nombre_clase = "Clase 0 (lóbulos oscuros)" if clase == 0 else "Clase 1 (fondo claro)"

    fig, axes = plt.subplots(
        1, 3,
        figsize=(19, 5.5),
        constrained_layout=True
    )

    mascara_visual = np.where(
        mascara_clases == clase,
        roi_gris,
        0
    )

    axes[0].imshow(mascara_visual, cmap="gray")
    axes[0].set_title(f"Región: {nombre_clase}", fontsize=11)
    axes[0].set_xlabel("X (píxeles)", fontsize=9)
    axes[0].set_ylabel("Y (píxeles)", fontsize=9)

    im = axes[1].imshow(
        r["glcm"],
        cmap="viridis",
        origin="lower",
        extent=[nivel_min, nivel_max, nivel_min, nivel_max]
    )

    axes[1].set_title(
        f"GLCM normalizada (θ=0°, d=1)\nResolución completa ({nivel_max - nivel_min + 1} niveles)",
        fontsize=11
    )
    axes[1].set_xlabel("Nivel de gris j", fontsize=9)
    axes[1].set_ylabel("Nivel de gris i", fontsize=9)

    fig.colorbar(im, ax=axes[1], shrink=0.8, label="p(i,j)")

    glcm_q = r["glcm_cuantizada"]

    im2 = axes[2].imshow(
        glcm_q,
        cmap="viridis",
        origin="lower"
    )

    axes[2].set_title(
        f"GLCM cuantizada ({NIVELES_GLCM} niveles)\ncon valores anotados",
        fontsize=11
    )
    axes[2].set_xlabel("Nivel de gris j (cuantizado)", fontsize=9)
    axes[2].set_ylabel("Nivel de gris i (cuantizado)", fontsize=9)

    etiquetas_bin = [f"{c:.0f}" for c in centros_bin]

    axes[2].set_xticks(range(NIVELES_GLCM))
    axes[2].set_xticklabels(etiquetas_bin, fontsize=7, rotation=90)
    axes[2].set_yticks(range(NIVELES_GLCM))
    axes[2].set_yticklabels(etiquetas_bin, fontsize=7)

    valor_max = glcm_q.max()

    for fila in range(NIVELES_GLCM):
        for columna in range(NIVELES_GLCM):

            valor = glcm_q[fila, columna]

            if valor > 0.001:

                color_texto = "white" if valor > valor_max * 0.5 else "black"

                axes[2].text(
                    columna, fila,
                    f"{valor:.3f}",
                    ha="center", va="center",
                    fontsize=6.5,
                    color=color_texto
                )

    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="p(i,j)")

    fig.suptitle(nombre_clase, fontsize=13)

    guardar_figura(fig)


pdf_radiomica.close()

print(f"\nPDF de características radiómicas generado en: {ruta_pdf_radiomica}")