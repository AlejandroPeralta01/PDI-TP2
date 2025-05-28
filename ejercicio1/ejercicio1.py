import cv2
import numpy as np
import matplotlib.pyplot as plt

def procesar_imagen_completo(imagen_path):
    """
    Procesa imagen mostrando TODOS los pasos intermedios
    """
    # 1. CARGAR IMAGEN ORIGINAL
    print("Paso 1: Cargando imagen original...")
    imagen_original = cv2.imread(imagen_path)
    if imagen_original is None:
        print("Error: No se pudo cargar la imagen")
        return None
    
    # 2. CONVERSIÓN A ESCALA DE GRISES
    print("Paso 2: Convirtiendo a escala de grises...")
    imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
    
    # 3. FILTRO GAUSSIANO (REDUCIR RUIDO)
    print("Paso 3: Aplicando filtro gaussiano...")
    imagen_blur = cv2.GaussianBlur(imagen_gris, (5, 5), 0)
    
    # 4. UMBRALIZACIÓN (OPCIONAL)
    print("Paso 4: Aplicando umbralización...")
    _, imagen_umbral = cv2.threshold(imagen_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # También probamos umbralización manual
    _, imagen_umbral_manual = cv2.threshold(imagen_blur, 127, 255, cv2.THRESH_BINARY)
    
    # 5. DETECCIÓN DE BORDES CON CANNY
    print("Paso 5: Detectando bordes con Canny...")
    # Probamos diferentes parámetros
    canny_30_100 = cv2.Canny(imagen_blur, 30, 100)
    canny_50_150 = cv2.Canny(imagen_blur, 50, 150)
    canny_70_200 = cv2.Canny(imagen_blur, 70, 200)
    
    # También Canny sobre imagen umbralizada
    canny_desde_umbral = cv2.Canny(imagen_umbral, 50, 150)
    
    # 6. OPERACIONES MORFOLÓGICAS
    print("Paso 6: Aplicando operaciones morfológicas...")
    
    # Diferentes kernels
    kernel_3x3 = np.ones((3,3), np.uint8)
    kernel_5x5 = np.ones((5,5), np.uint8)
    
    # Sobre la mejor imagen Canny (vamos a usar 50,150)
    canny_seleccionado = canny_50_150
    
    # Cerrar (close) - conecta líneas cercanas
    imagen_cerrada = cv2.morphologyEx(canny_seleccionado, cv2.MORPH_CLOSE, kernel_3x3, iterations=1)
    imagen_cerrada_2iter = cv2.morphologyEx(canny_seleccionado, cv2.MORPH_CLOSE, kernel_3x3, iterations=2)
    imagen_cerrada_3iter = cv2.morphologyEx(canny_seleccionado, cv2.MORPH_CLOSE, kernel_3x3, iterations=3)
    
    # Dilatación - expandir regiones blancas
    imagen_dilatada = cv2.dilate(imagen_cerrada_2iter, kernel_3x3, iterations=1)
    imagen_dilatada_2iter = cv2.dilate(imagen_cerrada_2iter, kernel_3x3, iterations=2)
    
    # Erosión - contraer regiones blancas
    imagen_erosionada = cv2.erode(imagen_dilatada, kernel_3x3, iterations=1)
    
    # 7. RELLENADO DE HUECOS
    print("Paso 7: Rellenando huecos...")
    
    # Método 1: FloodFill
    h, w = imagen_dilatada.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    imagen_floodfill = imagen_dilatada.copy()
    cv2.floodFill(imagen_floodfill, mask, (0,0), 255)
    imagen_floodfill_inv = cv2.bitwise_not(imagen_floodfill)
    imagen_rellena = imagen_dilatada | imagen_floodfill_inv
    
    # Método 2: Closing más agresivo
    imagen_rellena_closing = cv2.morphologyEx(imagen_dilatada, cv2.MORPH_CLOSE, kernel_5x5, iterations=3)
    
    # 8. COMPONENTES CONEXAS
    print("Paso 8: Analizando componentes conexas...")
    
    # Probamos sobre diferentes imágenes finales
    num_labels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(imagen_rellena, connectivity=8)
    num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(imagen_rellena_closing, connectivity=8)
    
    # 9. CREAR IMÁGENES DE COMPONENTES COLOREADAS
    print("Paso 9: Creando visualizaciones de componentes...")
    
    # Colorear componentes para visualización
    imagen_componentes1 = np.zeros((h, w, 3), dtype=np.uint8)
    imagen_componentes2 = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Asignar colores aleatorios a cada componente
    colores = np.random.randint(0, 255, size=(max(num_labels1, num_labels2), 3), dtype=np.uint8)
    colores[0] = [0, 0, 0]  # Fondo negro
    
    for i in range(num_labels1):
        imagen_componentes1[labels1 == i] = colores[i]
    
    for i in range(num_labels2):
        imagen_componentes2[labels2 == i] = colores[i]
    
    # 10. BOUNDING BOXES
    print("Paso 10: Dibujando bounding boxes...")
    
    imagen_bbox1 = imagen_original.copy()
    imagen_bbox2 = imagen_original.copy()
    
    # Dibujar bounding boxes (saltamos el componente 0 que es el fondo)
    componentes_validos1 = 0
    componentes_validos2 = 0
    
    for i in range(1, num_labels1):
        x = stats1[i, cv2.CC_STAT_LEFT]
        y = stats1[i, cv2.CC_STAT_TOP]
        w = stats1[i, cv2.CC_STAT_WIDTH]
        h = stats1[i, cv2.CC_STAT_HEIGHT]
        area = stats1[i, cv2.CC_STAT_AREA]
        
        # Filtrar componentes muy pequeños
        if area > 100:
            cv2.rectangle(imagen_bbox1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(imagen_bbox1, f'{i}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            componentes_validos1 += 1
    
    for i in range(1, num_labels2):
        x = stats2[i, cv2.CC_STAT_LEFT]
        y = stats2[i, cv2.CC_STAT_TOP]  
        w = stats2[i, cv2.CC_STAT_WIDTH]
        h = stats2[i, cv2.CC_STAT_HEIGHT]
        area = stats2[i, cv2.CC_STAT_AREA]
        
        if area > 100:
            cv2.rectangle(imagen_bbox2, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(imagen_bbox2, f'{i}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            componentes_validos2 += 1
    
    # ALMACENAR TODOS LOS RESULTADOS
    resultados = {
        'original': imagen_original,
        'gris': imagen_gris,
        'blur': imagen_blur,
        'umbral_otsu': imagen_umbral,
        'umbral_manual': imagen_umbral_manual,
        'canny_30_100': canny_30_100,
        'canny_50_150': canny_50_150,
        'canny_70_200': canny_70_200,
        'canny_desde_umbral': canny_desde_umbral,
        'cerrada_1iter': imagen_cerrada,
        'cerrada_2iter': imagen_cerrada_2iter,
        'cerrada_3iter': imagen_cerrada_3iter,
        'dilatada_1iter': imagen_dilatada,
        'dilatada_2iter': imagen_dilatada_2iter,
        'erosionada': imagen_erosionada,
        'rellena_floodfill': imagen_rellena,
        'rellena_closing': imagen_rellena_closing,
        'componentes_coloreados1': imagen_componentes1,
        'componentes_coloreados2': imagen_componentes2,
        'bbox_metodo1': imagen_bbox1,
        'bbox_metodo2': imagen_bbox2,
        'stats': {
            'componentes_metodo1': componentes_validos1,
            'componentes_metodo2': componentes_validos2,
            'total_labels1': num_labels1,
            'total_labels2': num_labels2
        }
    }
    
    print(f"\n=== ESTADÍSTICAS FINALES ===")
    print(f"Método 1 (FloodFill): {componentes_validos1} componentes válidos de {num_labels1-1} totales")
    print(f"Método 2 (Closing): {componentes_validos2} componentes válidos de {num_labels2-1} totales")
    
    return resultados

def mostrar_proceso_completo(resultados):
    """
    Muestra todas las imágenes del proceso en múltiples figuras
    """
    # FIGURA 1: PREPROCESAMIENTO
    fig1 = plt.figure(figsize=(20, 12))
    fig1.suptitle('PREPROCESAMIENTO', fontsize=16, fontweight='bold')
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(resultados['original'], cv2.COLOR_BGR2RGB))
    plt.title('1. Imagen Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(resultados['gris'], cmap='gray')
    plt.title('2. Escala de Grises')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(resultados['blur'], cmap='gray')
    plt.title('3. Filtro Gaussiano')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(resultados['umbral_otsu'], cmap='gray')
    plt.title('4a. Umbralización OTSU')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(resultados['umbral_manual'], cmap='gray')
    plt.title('4b. Umbralización Manual (127)')
    plt.axis('off')
    
    plt.tight_layout()
    
    # FIGURA 2: DETECCIÓN DE BORDES
    fig2 = plt.figure(figsize=(20, 12))
    fig2.suptitle('DETECCIÓN DE BORDES CON CANNY', fontsize=16, fontweight='bold')
    
    plt.subplot(2, 2, 1)
    plt.imshow(resultados['canny_30_100'], cmap='gray')
    plt.title('5a. Canny (30, 100)')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(resultados['canny_50_150'], cmap='gray')
    plt.title('5b. Canny (50, 150) ← SELECCIONADO')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(resultados['canny_70_200'], cmap='gray')
    plt.title('5c. Canny (70, 200)')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(resultados['canny_desde_umbral'], cmap='gray')
    plt.title('5d. Canny desde Umbralizada')
    plt.axis('off')
    
    plt.tight_layout()
    
    # FIGURA 3: OPERACIONES MORFOLÓGICAS
    fig3 = plt.figure(figsize=(20, 15))
    fig3.suptitle('OPERACIONES MORFOLÓGICAS', fontsize=16, fontweight='bold')
    
    plt.subplot(3, 3, 1)
    plt.imshow(resultados['canny_50_150'], cmap='gray')
    plt.title('Base: Canny (50,150)')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(resultados['cerrada_1iter'], cmap='gray')
    plt.title('6a. Cerrar (1 iteración)')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(resultados['cerrada_2iter'], cmap='gray')
    plt.title('6b. Cerrar (2 iteraciones)')
    plt.axis('off')
    
    plt.subplot(3, 3, 4)
    plt.imshow(resultados['cerrada_3iter'], cmap='gray')
    plt.title('6c. Cerrar (3 iteraciones)')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(resultados['dilatada_1iter'], cmap='gray')
    plt.title('6d. Dilatar (1 iter)')
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.imshow(resultados['dilatada_2iter'], cmap='gray')
    plt.title('6e. Dilatar (2 iter)')
    plt.axis('off')
    
    plt.subplot(3, 3, 7)
    plt.imshow(resultados['erosionada'], cmap='gray')
    plt.title('6f. Erosionar')
    plt.axis('off')
    
    plt.tight_layout()
    
    # FIGURA 4: RELLENADO Y COMPONENTES
    fig4 = plt.figure(figsize=(20, 12))
    fig4.suptitle('RELLENADO Y COMPONENTES CONEXAS', fontsize=16, fontweight='bold')
    
    plt.subplot(2, 3, 1)
    plt.imshow(resultados['rellena_floodfill'], cmap='gray')
    plt.title('7a. Relleno FloodFill')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(resultados['rellena_closing'], cmap='gray')
    plt.title('7b. Relleno Closing Agresivo')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(resultados['componentes_coloreados1'], cv2.COLOR_BGR2RGB))
    plt.title('8a. Componentes Método 1')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(resultados['componentes_coloreados2'], cv2.COLOR_BGR2RGB))
    plt.title('8b. Componentes Método 2')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(resultados['bbox_metodo1'], cv2.COLOR_BGR2RGB))
    plt.title(f'9a. Bounding Boxes M1\n({resultados["stats"]["componentes_metodo1"]} componentes)')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(resultados['bbox_metodo2'], cv2.COLOR_BGR2RGB))
    plt.title(f'9b. Bounding Boxes M2\n({resultados["stats"]["componentes_metodo2"]} componentes)')
    plt.axis('off')
    
    plt.tight_layout()
    
    plt.show()

def analizar_pcb_paso_a_paso(imagen_path):
    """
    Función principal que ejecuta y muestra todo el proceso
    """
    print("=== ANÁLISIS COMPLETO PASO A PASO ===")
    print("Procesando imagen y generando todas las etapas intermedias...\n")
    
    resultados = procesar_imagen_completo(imagen_path)
    
    if resultados is None:
        return None
    
    print("\nMostrando visualizaciones...")
    mostrar_proceso_completo(resultados)
    
    return resultados

def comparar_parametros_canny(imagen_path):
    """
    Función auxiliar para comparar diferentes parámetros de Canny
    """
    imagen = cv2.imread(imagen_path)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Diferentes combinaciones de parámetros
    parametros = [
        (20, 60), (30, 100), (50, 150), (70, 200),
        (100, 250), (20, 100), (50, 200), (30, 150)
    ]
    
    plt.figure(figsize=(16, 12))
    plt.suptitle('COMPARACIÓN DE PARÁMETROS CANNY', fontsize=16, fontweight='bold')
    
    for i, (low, high) in enumerate(parametros):
        canny = cv2.Canny(blur, low, high)
        plt.subplot(2, 4, i+1)
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
        # Análisis completo paso a paso
        resultados = analizar_pcb_paso_a_paso(ruta_imagen)
        
        # Opcional: comparar parámetros de Canny
        # comparar_parametros_canny(ruta_imagen)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Verifica que la ruta de la imagen sea correcta")