import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def ordenar_puntos_clockwise(puntos):
    """
    puntos: numpy.ndarray con forma (4, 2) con coordenadas (x, y)
    Retorna: numpy.ndarray con los 4 puntos ordenados en sentido horario,
             comenzando por el punto "superior izquierdo".
    """
    # Convertir a float32 para cálculos estables
    puntos = puntos.astype(np.float32)

    # Paso 1: encontrar el centroide del rectángulo
    centro = np.mean(puntos, axis=0)

    # Paso 2: calcular el ángulo de cada punto respecto al centroide
    def angulo(punto, centro):
        return np.arctan2(punto[1] - centro[1], punto[0] - centro[0])

    # Paso 3: ordenar por ángulo en sentido horario (restar π/2 para empezar desde arriba)
    puntos_ordenados = sorted(
        puntos,
        key=lambda p: angulo(p, centro)
    )

    # Reordenar para que empiece por el punto superior izquierdo
    # A veces el primer punto puede estar mal orientado, así que lo ajustamos:
    # Calcular distancias combinadas x+y para encontrar el punto "superior izquierdo"
    idx_sup_izq = np.argmin([p[0] + p[1] for p in puntos_ordenados])
    puntos_ordenados = np.roll(puntos_ordenados, -idx_sup_izq, axis=0)

    return puntos_ordenados.astype(puntos.dtype)

def transformar_resistencia(ruta_imagen):
    """
    Aplica transformación de perspectiva a una imagen de resistencia
    
    Args:
        ruta_imagen (str): Ruta completa del archivo de imagen
    
    Returns:
        numpy.ndarray: Imagen transformada o None si hay error
    """
    # Leer la imagen de la resistencia
    img = cv2.imread(ruta_imagen)
    
    if img is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return None
    
    # Convertir a espacio de color HSV para mejor detección del azul
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Definir rango de color azul en HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Crear máscara para regiones azules
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"Error: No se encontraron contornos azules en {ruta_imagen}")
        return None
    
    # Encontrar el contorno más grande (asumiendo que es el cuadrilátero azul)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Aproximar el contorno a un polígono
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Obtener los puntos de las esquinas y asegurar que tengamos solo 4 puntos
    if len(approx) > 4:
        # Si tenemos más de 4 puntos, tomar los 4 puntos que forman el área más grande
        hull = cv2.convexHull(approx)
        hull = cv2.approxPolyDP(hull, epsilon, True)
        corners = hull.reshape(4, 2)
    else:
        corners = approx.reshape(4, 2)
    
    # Obtener las dimensiones del rectángulo delimitador de la placa azul
    width = 2000
    height = 500
    
    # Definir puntos de destino para la transformación de perspectiva
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    
    # Ordenar puntos en sentido horario
    src_points = ordenar_puntos_clockwise(corners)
    
    # Calcular matriz de transformación de perspectiva
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Aplicar transformación de perspectiva
    result = cv2.warpPerspective(img, matrix, (width, height))
    
    return result

def procesar_todas_las_imagenes():
    """
    Procesa todas las imágenes .jpg de la carpeta 'imagenes' 
    y guarda las transformadas en 'imagenes_out'
    """
    # Crear carpeta de salida si no existe
    carpeta_salida = 'imagenes_out'
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        print(f"Carpeta '{carpeta_salida}' creada.")
    
    # Buscar todas las imágenes .jpg en la carpeta 'imagenes'
    patron_busqueda = os.path.join('imagenes', '*.jpg')
    rutas_imagenes = glob.glob(patron_busqueda)
    
    if not rutas_imagenes:
        print("No se encontraron imágenes .jpg en la carpeta 'imagenes'")
        return
    
    print(f"Se encontraron {len(rutas_imagenes)} imágenes para procesar...")
    
    imagenes_procesadas = 0
    imagenes_fallidas = 0
    
    for ruta_imagen in rutas_imagenes:
        # Obtener el nombre del archivo sin la extensión
        nombre_archivo = os.path.basename(ruta_imagen)
        nombre_sin_extension = os.path.splitext(nombre_archivo)[0]
        
        print(f"Procesando: {nombre_archivo}...")
        
        # Aplicar transformación
        imagen_transformada = transformar_resistencia(ruta_imagen)
        
        if imagen_transformada is not None:
            # Crear nombre de archivo de salida
            nombre_salida = f"{nombre_sin_extension}_out.jpg"
            ruta_salida = os.path.join(carpeta_salida, nombre_salida)
            
            # Guardar imagen transformada
            exito = cv2.imwrite(ruta_salida, imagen_transformada)
            
            if exito:
                print(f"  Guardada como: {nombre_salida}")
                imagenes_procesadas += 1
            else:
                print(f"  Error al guardar: {nombre_salida}")
                imagenes_fallidas += 1
        else:
            print(f"   Error al procesar: {nombre_archivo}")
            imagenes_fallidas += 1
    
    print(f"\n=== RESUMEN ===")
    print(f"Imágenes procesadas exitosamente: {imagenes_procesadas}")
    print(f"Imágenes que fallaron: {imagenes_fallidas}")
    print(f"Total de imágenes: {len(rutas_imagenes)}")

# Ejecutar el procesamiento
if __name__ == "__main__":
    procesar_todas_las_imagenes()