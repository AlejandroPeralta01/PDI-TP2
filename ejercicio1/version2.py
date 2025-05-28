import cv2
import numpy as np
import matplotlib.pyplot as plt

def procesar_imagen_pcb(imagen_path):
    """
    Procesa una imagen de PCB para detectar componentes usando solo OpenCV
    """
    # 1. CARGA Y PREPROCESAMIENTO
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        print("Error: No se pudo cargar la imagen")
        return None, None, None, None
    
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque gaussiano para reducir ruido
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # 2. DETECCIÓN DE BORDES CON CANNY
    canny = cv2.Canny(blur, 50, 150)
    
    # 3. OPERACIONES MORFOLÓGICAS
    kernel = np.ones((3,3), np.uint8)
    
    # Cerrar contornos
    canny_cerrado = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Dilatar para conectar componentes cercanos
    dilatado = cv2.dilate(canny_cerrado, kernel, iterations=1)
    
    # Rellenar huecos usando floodFill
    h, w = dilatado.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    im_floodfill = dilatado.copy()
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    imagen_binaria = dilatado | im_floodfill_inv
    
    return imagen, gris, canny, imagen_binaria

def detectar_componentes_cv2(imagen_binaria):
    """
    Detecta componentes usando cv2.connectedComponentsWithStats
    """
    # 4. COMPONENTES CONEXAS CON ESTADÍSTICAS
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_binaria, connectivity=8)
    
    componentes = {
        'resistencias': [],
        'capacitores': [],
        'chips': [],
        'otros': []
    }
    
    # Saltamos el label 0 (fondo)
    for i in range(1, num_labels):
        # Extraer estadísticas del componente
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filtrar componentes muy pequeños (ruido)
        if area < 100:
            continue
        
        # Calcular propiedades geométricas
        relacion_aspecto = max(w, h) / min(w, h)
        
        # Crear máscara del componente para análisis adicional
        mask_componente = (labels == i).astype(np.uint8) * 255
        
        # Calcular solidez (área del componente / área del rectángulo envolvente)
        solidez = area / (w * h)
        
        # Crear diccionario con información del componente
        componente_info = {
            'id': i,
            'bbox': (x, y, w, h),
            'area': area,
            'relacion_aspecto': relacion_aspecto,
            'solidez': solidez,
            'centroide': centroids[i],
            'mask': mask_componente
        }
        
        # 5. CLASIFICACIÓN
        if clasificar_resistencia(area, relacion_aspecto, solidez):
            componentes['resistencias'].append(componente_info)
        elif clasificar_capacitor(area, relacion_aspecto, solidez):
            componentes['capacitores'].append(componente_info)
        elif clasificar_chip(area, relacion_aspecto, solidez):
            componentes['chips'].append(componente_info)
        else:
            componentes['otros'].append(componente_info)
    
    return componentes, labels

def clasificar_resistencia(area, relacion_aspecto, solidez):
    """
    Criterios para identificar resistencias
    """
    # Resistencias: alargadas, tamaño mediano, forma compacta
    return (200 < area < 2000 and 
            2.5 < relacion_aspecto < 8 and
            solidez > 0.6)

def clasificar_capacitor(area, relacion_aspecto, solidez):
    """
    Criterios para identificar capacitores
    """
    # Capacitores: más cuadrados/circulares, varios tamaños
    return (300 < area < 5000 and 
            relacion_aspecto < 3 and
            solidez > 0.5)

def clasificar_chip(area, relacion_aspecto, solidez):
    """
    Criterios para identificar chips
    """
    # Chips: rectangulares, grandes, muy compactos
    return (area > 1500 and 
            1 < relacion_aspecto < 2.5 and
            solidez > 0.7)

def clasificar_capacitores_por_tamaño(capacitores):
    """
    Clasifica capacitores en pequeños, medianos y grandes
    """
    if not capacitores:
        return {'pequeños': [], 'medianos': [], 'grandes': []}
    
    # Obtener todas las áreas
    areas = [cap['area'] for cap in capacitores]
    
    # Calcular umbrales usando percentiles
    areas_sorted = sorted(areas)
    n = len(areas_sorted)
    
    if n == 1:
        return {'pequeños': [], 'medianos': capacitores, 'grandes': []}
    elif n == 2:
        return {'pequeños': [capacitores[0]], 'medianos': [], 'grandes': [capacitores[1]]}
    
    # Usar terciles para clasificación
    umbral_1 = areas_sorted[n//3]
    umbral_2 = areas_sorted[2*n//3]
    
    clasificacion = {'pequeños': [], 'medianos': [], 'grandes': []}
    
    for cap in capacitores:
        if cap['area'] <= umbral_1:
            clasificacion['pequeños'].append(cap)
        elif cap['area'] <= umbral_2:
            clasificacion['medianos'].append(cap)
        else:
            clasificacion['grandes'].append(cap)
    
    return clasificacion

def crear_imagen_segmentada(imagen_original, componentes):
    """
    Crea imagen mostrando todos los componentes detectados
    """
    imagen_resultado = imagen_original.copy()
    
    # Colores para cada tipo (BGR)
    colores = {
        'resistencias': (0, 255, 0),    # Verde
        'capacitores': (0, 0, 255),     # Rojo
        'chips': (255, 0, 0),           # Azul
        'otros': (0, 255, 255)          # Amarillo
    }
    
    for tipo, lista_componentes in componentes.items():
        color = colores[tipo]
        for comp in lista_componentes:
            x, y, w, h = comp['bbox']
            
            # Dibujar rectángulo
            cv2.rectangle(imagen_resultado, (x, y), (x+w, y+h), color, 2)
            
            # Añadir etiqueta
            cv2.putText(imagen_resultado, tipo[:4], 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
    
    return imagen_resultado

def crear_imagen_capacitores_clasificados(imagen_original, caps_clasificados):
    """
    Crea imagen mostrando capacitores clasificados por tamaño
    """
    imagen_caps = imagen_original.copy()
    
    # Colores para tamaños (BGR)
    colores_tamaño = {
        'pequeños': (255, 255, 0),    # Cyan
        'medianos': (255, 0, 255),    # Magenta
        'grandes': (0, 165, 255)      # Naranja
    }
    
    for tamaño, caps in caps_clasificados.items():
        color = colores_tamaño[tamaño]
        for cap in caps:
            x, y, w, h = cap['bbox']
            
            # Dibujar rectángulo
            cv2.rectangle(imagen_caps, (x, y), (x+w, y+h), color, 3)
            
            # Añadir etiqueta de tamaño
            cv2.putText(imagen_caps, tamaño[:3], 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
    
    return imagen_caps

def analizar_pcb_cv2(imagen_path):
    """
    Función principal usando solo OpenCV
    """
    print("=== ANÁLISIS DE PLACA PCB (OpenCV) ===")
    
    # a) Procesar imagen
    print("1. Procesando imagen...")
    resultado_procesamiento = procesar_imagen_pcb(imagen_path)
    if resultado_procesamiento[0] is None:
        return None, None
    
    imagen_orig, gris, canny, binaria = resultado_procesamiento
    
    # Detectar componentes usando connectedComponentsWithStats
    print("2. Detectando componentes con connectedComponentsWithStats...")
    componentes, labels = detectar_componentes_cv2(binaria)
    
    # b) Clasificar capacitores por tamaño
    print("3. Clasificando capacitores por tamaño...")
    caps_clasificados = clasificar_capacitores_por_tamaño(componentes['capacitores'])
    
    # c) Contar resistencias
    num_resistencias = len(componentes['resistencias'])
    
    # Mostrar resultados por consola
    print("\n=== RESULTADOS ===")
    print(f"Resistencias detectadas: {num_resistencias}")
    print(f"Capacitores detectados: {len(componentes['capacitores'])}")
    print(f"  - Pequeños: {len(caps_clasificados['pequeños'])}")
    print(f"  - Medianos: {len(caps_clasificados['medianos'])}")
    print(f"  - Grandes: {len(caps_clasificados['grandes'])}")
    print(f"Chips detectados: {len(componentes['chips'])}")
    print(f"Otros componentes: {len(componentes['otros'])}")
    
    # Crear imágenes de resultado
    imagen_segmentada = crear_imagen_segmentada(imagen_orig, componentes)
    imagen_caps_clasificados = crear_imagen_capacitores_clasificados(imagen_orig, caps_clasificados)
    
    # Visualizar resultados
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(imagen_orig, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(gris, cmap='gray')
    plt.title('Escala de Grises')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(canny, cmap='gray')
    plt.title('Detección Canny')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(binaria, cmap='gray')
    plt.title('Imagen Binaria Final')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(imagen_segmentada, cv2.COLOR_BGR2RGB))
    plt.title('Componentes Detectados')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(imagen_caps_clasificados, cv2.COLOR_BGR2RGB))
    plt.title('Capacitores por Tamaño')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return componentes, caps_clasificados

# FUNCIÓN PARA AJUSTAR PARÁMETROS
def ajustar_parametros_canny(imagen_path):
    """
    Función auxiliar para ajustar parámetros de Canny interactivamente
    """
    imagen = cv2.imread(imagen_path)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Probar diferentes umbrales
    umbrales = [(30, 100), (50, 150), (70, 200), (100, 250)]
    
    plt.figure(figsize=(12, 8))
    for i, (low, high) in enumerate(umbrales):
        canny = cv2.Canny(blur, low, high)
        plt.subplot(2, 2, i+1)
        plt.imshow(canny, cmap='gray')
        plt.title(f'Canny ({low}, {high})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# EJEMPLO DE USO
if __name__ == "__main__":
    # Reemplazar con la ruta de tu imagen
    ruta_imagen = "placa.png"
    
    try:
        # Opcional: ajustar parámetros primero
        # ajustar_parametros_canny(ruta_imagen)
        
        # Análisis principal
        componentes, caps_clasificados = analizar_pcb_cv2(ruta_imagen)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Verifica que la ruta de la imagen sea correcta")