import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Diccionario de valores de colores para resistencias
VALORES_COLORES = {
    "Negro": 0,
    "Marron": 1,
    "Rojo": 2,
    "Naranja": 3,
    "Amarillo": 4,
    "Verde": 5,
    "Azul": 6,
    "Violeta": 7,
    "Gris": 8,
    "Blanco": 9
}

# Multiplicadores para la tercera banda
MULTIPLICADORES = {
    "Negro": 1,
    "Marron": 10,
    "Rojo": 100,
    "Naranja": 1000,
    "Amarillo": 10000,
    "Verde": 100000,
    "Azul": 1000000,
    "Violeta": 10000000,
    "Gris": 100000000,
    "Blanco": 1000000000
}

def get_color_name(hsv):
    """
    Función para clasificar los colores de forma aproximada
    """
    h, s, v = hsv

    # Bajo nivel de saturación y valor alto = blanco / gris claro
    if s < 65 and v > 150:
        return "Blanco"
    elif v < 50:
        return "Negro"
    elif s < 60:
        return "Gris"
    
    if h < 10 and v < 120:
        return "Marron"
    elif h < 10 or h >= 170:
        return "Rojo"
    elif 10 <= h < 20:
        return "Naranja"
    elif 20 <= h < 35:
        return "Amarillo"
    elif 35 <= h < 85:
        return "Verde"
    elif 85 <= h < 130:
        return "Azul"
    elif 130 <= h < 170:
        return "Violeta"
    else:
        return "Desconocido"

def rotar_si_es_necesario(img, hsv_img, contornos):
    """
    Detecta si el área más grande está a la izquierda y espeja la imagen si es necesario.
    Luego recalcula los contornos en la imagen espejada.
    """
    # Ordenar contornos por coordenada x
    contornos_ordenados = sorted(contornos, key=lambda c: cv2.boundingRect(c)[0])
    
    # Calcular áreas de todos los contornos
    areas = [cv2.contourArea(c) for c in contornos_ordenados]
    
    # Inicialmente no rotada
    imagen_resultado = img.copy()
    hsv_resultado = hsv_img.copy()
    contornos_resultantes = contornos_ordenados
    
    # Si el contorno más a la izquierda tiene el área máxima, espejar horizontalmente
    if areas and areas[0] == max(areas):
        print("Imagen espejada porque el mayor contorno estaba a la izquierda.")
        
        # Espejar la imagen
        imagen_resultado = cv2.flip(img, 1)
        hsv_resultado = cv2.flip(hsv_img, 1)
        
        # Recalcular todo el proceso de detección de contornos con la imagen espejada
        # 1. Preprocesamiento en imagen espejada
        blur = cv2.GaussianBlur(imagen_resultado, (3, 3), 0)
        
        # 2. Máscara para eliminar fondo (cuerpo amarillo de la resistencia)
        lower_yellow = np.array([14, 100, 100])
        upper_yellow = np.array([18, 200, 200])
        mask_fondo = cv2.inRange(blur, lower_yellow, upper_yellow)
        
        # 3. Invertir para quedarse las bandas
        mask_bandas = cv2.bitwise_not(mask_fondo)
        
        # 4. Limpiar con morfología
        kernel = np.ones((17, 5), np.uint8)
        mask_bandas_limpia = cv2.morphologyEx(mask_bandas, cv2.MORPH_CLOSE, kernel)
        mask_bandas_limpia = cv2.morphologyEx(mask_bandas_limpia, cv2.MORPH_OPEN, kernel)
        
        # 5. Invertir para encontrar contornos
        img_invertida = cv2.bitwise_not(mask_bandas_limpia)
        
        # 6. Recalcular contornos en la imagen espejada
        contornos_nuevos, _ = cv2.findContours(img_invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_resultantes = sorted(contornos_nuevos, key=lambda c: cv2.boundingRect(c)[0])
        
    else:
        print("No fue necesario espejar.")
    
    return imagen_resultado, hsv_resultado, contornos_resultantes

def calcular_valor_resistencia(colores_detectados):
    """
    Calcula el valor de la resistencia basado en los colores detectados
    """
    if len(colores_detectados) < 3:
        return "Insuficientes colores detectados", 0
    
    # Tomar los primeros 3 colores
    color1, color2, color3 = colores_detectados[:3]
    
    # Verificar que los colores estén en el diccionario
    if color1 not in VALORES_COLORES or color2 not in VALORES_COLORES or color3 not in MULTIPLICADORES:
        return "Colores no reconocidos", 0
    
    # Calcular valor
    digito1 = VALORES_COLORES[color1]
    digito2 = VALORES_COLORES[color2]
    multiplicador = MULTIPLICADORES[color3]
    
    valor = (digito1 * 10 + digito2) * multiplicador
    
    # Formatear el valor para mejor legibilidad
    if valor >= 1000000:
        valor_formateado = f"{valor/1000000:.1f}MΩ"
    elif valor >= 1000:
        valor_formateado = f"{valor/1000:.1f}kΩ"
    else:
        valor_formateado = f"{valor}Ω"
    
    return valor_formateado, valor

def dibujar_separadores_entre_contornos(img_base, contornos):
    """
    Dibuja líneas separadoras entre contornos detectados
    """
    img_separadores = img_base.copy()

    # Obtener bounding boxes y ordenarlos por coordenada x
    bounding_boxes = [cv2.boundingRect(c) for c in contornos]
    bounding_boxes.sort(key=lambda b: b[0])

    if not bounding_boxes:
        return img_separadores

    # Obtener el primer bounding box (el de más a la izquierda)
    x_first, y_first, w_first, h_first = bounding_boxes[0]
    y_medio_primer_contorno = y_first + h_first // 2

    # Línea horizontal de referencia
    cv2.line(img_separadores, (0, y_medio_primer_contorno),
             (img_separadores.shape[1], y_medio_primer_contorno), (0, 255, 0), 3)
    
    # Dibujar líneas entre cada par de bounding boxes consecutivos
    for i in range(len(bounding_boxes) - 1):
        x1, _, w1, _ = bounding_boxes[i]
        x2, _, _, _ = bounding_boxes[i + 1]
        x_medio = (x1 + w1 + x2) // 2

        # Dibujar línea vertical en el centro entre los dos contornos
        cv2.line(img_separadores, (x_medio, 0), (x_medio, img_separadores.shape[0]), (0, 0, 255), 3)

    return img_separadores

def detectar_color_en_intersecciones(img_base, hsv_img, contornos):
    """
    Detecta colores en las intersecciones entre bandas
    """
    img_resultado = img_base.copy()
    colores_detectados = []
    
    # Obtener bounding boxes y ordenarlos por x
    bounding_boxes = [cv2.boundingRect(c) for c in contornos]
    bounding_boxes.sort(key=lambda b: b[0])

    if not bounding_boxes:
        return img_resultado, colores_detectados

    # Línea horizontal: altura media del primer contorno
    x_first, y_first, w_first, h_first = bounding_boxes[0]
    y_medio = y_first + h_first // 2
    indice_medio = (len(bounding_boxes) - 1) // 2

    # Iterar entre cada par de contornos consecutivos
    for i in range(len(bounding_boxes) - 1):
        x1, _, w1, _ = bounding_boxes[i]
        x2, _, _, _ = bounding_boxes[i + 1]
        x_medio = (x1 + w1 + x2) // 2

        # Región pequeña alrededor de la intersección
        ancho = 4
        alto = 4
        region = hsv_img[y_medio - alto//2 : y_medio + alto//2,
                         x_medio - ancho//2 : x_medio + ancho//2]

        if region.size == 0:
            continue

        # K-Means
        Z = region.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(label.flatten())
        dominant_hsv = center[np.argmax(counts)].astype(int)

        # Clasificar color
        color_name = get_color_name(dominant_hsv)
        colores_detectados.append(color_name)
        
        # Flechas
        largo_flecha = 100 if i == indice_medio else 50
        texto_offset = 30 if i == indice_medio else 15

        # Dibujar flecha
        cv2.arrowedLine(img_resultado,
                        (x_medio, y_medio - largo_flecha),
                        (x_medio, y_medio - 2),
                        (0, 255, 255), 2, tipLength=0.3)

        # Agregar texto
        cv2.putText(img_resultado, color_name,
                    (x_medio - 30, y_medio - largo_flecha - texto_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return img_resultado, colores_detectados

def procesar_imagen_resistencia(ruta_imagen):
    """
    Procesa una imagen de resistencia y devuelve todas las etapas del procesamiento
    """
    # Cargar imagen
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        print(f"Error: No se pudo cargar {ruta_imagen}")
        return None
    
    img = img_bgr.copy()
    
    # 1. Imagen original
    img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Preprocesamiento
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # 3. Máscara para eliminar fondo (cuerpo amarillo de la resistencia)
    lower_yellow = np.array([14, 100, 100])
    upper_yellow = np.array([18, 200, 200])
    mask_fondo = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 4. Invertir para quedarse con las bandas
    mask_bandas = cv2.bitwise_not(mask_fondo)
    
    # 5. Limpiar con morfología
    kernel = np.ones((17, 5), np.uint8)
    mask_bandas_limpia = cv2.morphologyEx(mask_bandas, cv2.MORPH_CLOSE, kernel)
    mask_bandas_limpia = cv2.morphologyEx(mask_bandas_limpia, cv2.MORPH_OPEN, kernel)
    
    # 6. Invertir para encontrar contornos
    img_invertida = cv2.bitwise_not(mask_bandas_limpia)
    
    # 7. Detección de contornos
    contornos, _ = cv2.findContours(img_invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 8. Rotar si es necesario y recalcular contornos
    img_final, hsv_final, contornos_finales = rotar_si_es_necesario(img, hsv, contornos)
    
    # 9. Imagen con contornos dibujados (usando la imagen final)
    img_contornos = img_final.copy()
    cv2.drawContours(img_contornos, contornos_finales, -1, (0, 255, 0), 2)
    img_contornos_rgb = cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB)
    
    # 10. Imagen con separadores
    img_separadores = dibujar_separadores_entre_contornos(img_final, contornos_finales)
    img_separadores_rgb = cv2.cvtColor(img_separadores, cv2.COLOR_BGR2RGB)
    
    # 11. Imagen con colores detectados y cálculo de resistencia
    img_colores, colores_detectados = detectar_color_en_intersecciones(img_final, hsv_final, contornos_finales)
    img_colores_rgb = cv2.cvtColor(img_colores, cv2.COLOR_BGR2RGB)
    
    # 12. Calcular valor de resistencia
    valor_resistencia, valor_numerico = calcular_valor_resistencia(colores_detectados)
    
    return {
        'original': img_original,
        'hsv': hsv_final,  
        'mask_fondo': mask_fondo,
        'mask_bandas': mask_bandas,
        'mask_limpia': mask_bandas_limpia,
        'invertida': img_invertida,
        'contornos': img_contornos_rgb,
        'separadores': img_separadores_rgb,
        'colores': img_colores_rgb,
        'num_contornos': len(contornos_finales),
        'colores_detectados': colores_detectados,
        'valor_resistencia': valor_resistencia,
        'valor_numerico': valor_numerico
    }

def crear_imagen_compuesta(resultados, nombre_imagen):
    """
    Crea una imagen compuesta con todos los pasos del procesamiento en formato vertical
    """
    # Crear figura vertical con 9 subplots (una fila por imagen)
    fig, axes = plt.subplots(9, 1, figsize=(12, 26))
    fig.suptitle(f'Procesamiento de {nombre_imagen}', fontsize=20, y=0.995)
    
    # Títulos para cada subplot
    titulos = [
        '1. Imagen Original',
        '2. Imagen HSV', 
        '3. Máscara del Fondo (Amarillo)',
        '4. Máscara de las Bandas',
        '5. Máscara Limpia (Morfología)',
        '6. Imagen Invertida',
        '7. Contornos Detectados',
        '8. Separadores entre Bandas',
        '9. Colores Detectados'
    ]
    
    # Imágenes a mostrar
    imagenes = [
        resultados['original'],
        resultados['hsv'],
        resultados['mask_fondo'],
        resultados['mask_bandas'],
        resultados['mask_limpia'],
        resultados['invertida'],
        resultados['contornos'],
        resultados['separadores'],
        resultados['colores']
    ]
    
    # Configurar cada subplot
    for i, (ax, titulo, imagen) in enumerate(zip(axes, titulos, imagenes)):
        if len(imagen.shape) == 3 and imagen.shape[2] == 3:
            ax.imshow(imagen)
        else:
            ax.imshow(imagen, cmap='gray')
        
        # Título más grande y con mejor formato
        ax.set_title(titulo, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
    
    # Información adicional en la parte inferior
    info_text = f"""Contornos encontrados: {resultados["num_contornos"]}
                    Colores detectados: {' → '.join(resultados["colores_detectados"])}
                    Valor calculado: {resultados["valor_resistencia"]}"""
    
    fig.text(0.5, 0.02, info_text, 
             fontsize=12, ha='center', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    # Ajustar espaciado entre subplots
    plt.subplots_adjust(top=0.97, bottom=0.08, hspace=0.3)
    
    return fig

def procesar_todas_las_resistencias():
    """
    Procesa todas las imágenes que terminan en '_a_out.jpg' de la carpeta imagenes_out
    """
    # Crear carpeta de salida si no existe
    carpeta_salida = 'imagenes_intermedias'
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        print(f"Carpeta '{carpeta_salida}' creada.")
    
    # Buscar todas las imágenes que terminan en '_a_out.jpg'
    patron_busqueda = os.path.join('imagenes_out', '*_a_out.jpg')
    rutas_imagenes = glob.glob(patron_busqueda)
    
    if not rutas_imagenes:
        print("No se encontraron imágenes '*_a_out.jpg' en la carpeta 'imagenes_out'")
        return
    
    print(f"Se encontraron {len(rutas_imagenes)} imágenes para procesar...")
    
    # Crear archivo de resumen
    resumen_path = os.path.join(carpeta_salida, 'resumen_resistencias.txt')
    with open(resumen_path, 'w', encoding='utf-8') as archivo_resumen:
        archivo_resumen.write("=== RESUMEN DE RESISTENCIAS PROCESADAS ===\n\n")
        
        for ruta_imagen in rutas_imagenes:
            # Obtener nombre del archivo
            nombre_archivo = os.path.basename(ruta_imagen)
            nombre_sin_extension = os.path.splitext(nombre_archivo)[0]
            
            print(f"Procesando: {nombre_archivo}...")
            
            # Procesar imagen
            resultados = procesar_imagen_resistencia(ruta_imagen)
            
            if resultados is not None:
                # Crear imagen compuesta
                fig = crear_imagen_compuesta(resultados, nombre_sin_extension)
                
                # Guardar imagen compuesta
                nombre_salida = f"{nombre_sin_extension}_procesamiento.png"
                ruta_salida = os.path.join(carpeta_salida, nombre_salida)
                
                fig.savefig(ruta_salida, dpi=150, bbox_inches='tight')
                plt.close(fig)  # Cerrar figura para liberar memoria
                
                # Escribir en el resumen
                archivo_resumen.write(f"Archivo: {nombre_archivo}\n")
                archivo_resumen.write(f"Colores: {' → '.join(resultados['colores_detectados'])}\n")
                archivo_resumen.write(f"Valor: {resultados['valor_resistencia']}\n")
                archivo_resumen.write(f"Contornos: {resultados['num_contornos']}\n")
                archivo_resumen.write("-" * 50 + "\n\n")
                
                print(f"  Guardada como: {nombre_salida}")
                print(f"  Valor: {resultados['valor_resistencia']}")
            else:
                print(f"  Error al procesar: {nombre_archivo}")
                archivo_resumen.write(f"Archivo: {nombre_archivo} - ERROR AL PROCESAR\n")
                archivo_resumen.write("-" * 50 + "\n\n")
    
    print(f"\n=== PROCESAMIENTO COMPLETADO ===")
    print(f"Las imágenes procesadas se guardaron en la carpeta '{carpeta_salida}'")
    print(f"Resumen guardado en: {resumen_path}")

# Ejecutar el procesamiento
if __name__ == "__main__":
    procesar_todas_las_resistencias()