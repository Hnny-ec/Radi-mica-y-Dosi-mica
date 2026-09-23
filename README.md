# Radiómica y Dosiómica

El presente repositorio contiene los algoritmos utilizados para la resolución de la evaluación del módulo 4.

## Scripts

- **Dimensión.py**: Lee la imagen utilizada, en este caso `Imagen.JPEG`, para obtener la ROI de manera manual.
- **ROI.py**: Permite verificar de forma manual que los datos de la ROI ingresados corresponden a la sección a estudiar.
- **Histograma.py**: Genera el histograma de la ROI.
- **Máscaras.py**: Aplica filtros gaussianos con distintos tamaños de máscara y desviaciones estándar sobre la ROI, y calcula cómo varía su entropía respecto a la imagen original.
- **Kmeans.py**: Segmenta la ROI en dos clases mediante el algoritmo k-means, probando distintas condiciones iniciales y comparando sus resultados (iteraciones, varianza intra/entre clases).
- **radiomica.py**: Calcula características radiómicas (primer orden y matriz GLCM) para cada clase segmentada con k-means.
