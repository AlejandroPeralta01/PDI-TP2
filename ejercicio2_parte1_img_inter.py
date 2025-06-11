import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------
# FUNCIONES
# ------------------------------

# Función para ordenar los puntos de un cuadrilátero en sentido horario
def ordenar_puntos_clockwise(puntos):
    puntos = puntos.astype(np.float32)
    centro = np.mean(puntos, axis=0)
    def angulo(punto, centro):
        return np.arctan2(punto[1] - centro[1], punto[0] - centro[0])
    puntos_ordenados = sorted(puntos, key=lambda p: angulo(p, centro))
    idx_sup_izq = np.argmin([p[0] + p[1] for p in puntos_ordenados])
    puntos_ordenados = np.roll(puntos_ordenados, -idx_sup_izq, axis=0)
    return puntos_ordenados.astype(puntos.dtype)

# ------------------------------
# PROCESAMIENTO DE IMAGEN
# ------------------------------

# Cargar imagen desde la carpeta especificada
img_path = os.path.join('imagenes', 'R1_d.jpg')
img = cv2.imread(img_path)

# Verificar si la imagen se cargó correctamente
if img is None:
    print(f"Error: No se pudo cargar la imagen desde {img_path}")
    print("Verifica que el archivo exista en la ruta especificada.")
    exit()

print(f"Imagen cargada exitosamente desde: {img_path}")
print(f"Dimensiones de la imagen: {img.shape}")

# Convertir a HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Crear máscara azul
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

# Contornos
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("No se encontraron contornos azules.")
    exit()

# Contorno más grande
largest_contour = max(contours, key=cv2.contourArea)

# Imagen con contorno sin aproximar
img_contours_raw = img.copy()
cv2.drawContours(img_contours_raw, [largest_contour], -1, (0, 255, 255), 3)

# Aproximación del contorno
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# Extraer esquinas
if len(approx) > 4:
    hull = cv2.convexHull(approx)
    hull = cv2.approxPolyDP(hull, epsilon, True)
    corners = hull.reshape(4, 2)
else:
    corners = approx.reshape(4, 2)

# Imagen con contorno final
img_contours_final = img.copy()
cv2.drawContours(img_contours_final, [approx], -1, (0, 255, 0), 3)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_raw_rgb = cv2.cvtColor(img_contours_raw, cv2.COLOR_BGR2RGB)
img_final_rgb = cv2.cvtColor(img_contours_final, cv2.COLOR_BGR2RGB)

# ------------------------------
# TRANSFORMACIÓN DE PERSPECTIVA
# ------------------------------

width = 2000
height = 500
dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
src_points = ordenar_puntos_clockwise(corners)
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_warped = cv2.warpPerspective(img, matrix, (width, height))

# ------------------------------
# GUARDAR RESULTADOS
# ------------------------------

# Crear carpeta de salida
output_folder = 'proceso_ej2_parte1'
os.makedirs(output_folder, exist_ok=True)

# Guardar imágenes con nombres enumerados
cv2.imwrite(os.path.join(output_folder, 'paso1_imagen_original.jpg'), img)
plt.imsave(os.path.join(output_folder, 'paso2_imagen_hsv.jpg'), img_hsv)
cv2.imwrite(os.path.join(output_folder, 'paso3_mascara_binaria.jpg'), mask)
cv2.imwrite(os.path.join(output_folder, 'paso4_contorno_sin_aproximar.jpg'), img_contours_raw)
cv2.imwrite(os.path.join(output_folder, 'paso5_contorno_aproximado.jpg'), img_contours_final)
cv2.imwrite(os.path.join(output_folder, 'paso6_imagen_transformada.jpg'), img_warped)

print(f"Resultados guardados en la carpeta: {output_folder}")
print("Archivos generados:")
print("- paso1_imagen_original.jpg")
print("- paso2_imagen_hsv.jpg")
print("- paso3_mascara_binaria.jpg")
print("- paso4_contorno_sin_aproximar.jpg")
print("- paso5_contorno_aproximado.jpg")
print("- paso6_imagen_transformada.jpg")