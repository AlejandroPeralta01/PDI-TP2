import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_resistencias_parejas(ruta_imagen):
    """
    Función para detectar resistencias mostrando todas las etapas del proceso
    en 4 figuras separadas con pares de imágenes (2x2 cada una)
    """
    
    # Cargar y convertir imagen original
    imagen = cv2.imread(ruta_imagen)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    
    # ETAPA 2: Conversión a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # ETAPA 3: Aplicación de filtro Gaussiano para suavizar bordes
    f_blur = cv2.GaussianBlur(gris, (11, 11), 1)
    
    # ETAPA 4: Detección de bordes con algoritmo Canny
    canny = cv2.Canny(f_blur, 50, 80)
    
    # ETAPA 5: Operación morfológica de clausura
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    canny_closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    
    # ETAPA 6: Dilatación de las líneas para engrosarlas
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    canny_dilated = cv2.dilate(canny_closed, kernel_dilate, iterations=1)
    
    # ETAPA 7: Análisis de componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(canny_dilated, connectivity=8)
    
    # Visualizar todas las componentes con colores aleatorios
    labels_colored = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Fondo negro
    
    for i in range(num_labels):
        labels_colored[labels == i] = colors[i]
    
    # FIGURA 1: Etapas 1 y 2 (Original y Escala de Grises)
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 8))
    fig1.suptitle('Proceso de Detección - Etapas 1 y 2', fontsize=16, fontweight='bold')
    
    axes1[0].imshow(imagen_rgb)
    axes1[0].set_title('1. Imagen Original de la Placa Electrónica', fontsize=14, fontweight='bold')
    axes1[0].axis('off')
    
    axes1[1].imshow(gris, cmap='gray')
    axes1[1].set_title('2. Conversión a Escala de Grises', fontsize=14, fontweight='bold')
    axes1[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # FIGURA 2: Etapas 3 y 4 (Filtro Gaussiano y Canny)
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))
    fig2.suptitle('Proceso de Detección - Etapas 3 y 4', fontsize=16, fontweight='bold')
    
    axes2[0].imshow(f_blur, cmap='gray')
    axes2[0].set_title('3. Filtro Gaussiano para Suavizar Bordes', fontsize=14, fontweight='bold')
    axes2[0].axis('off')
    
    axes2[1].imshow(canny, cmap='gray')
    axes2[1].set_title('4. Detección de Bordes (Algoritmo Canny)', fontsize=14, fontweight='bold')
    axes2[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # FIGURA 3: Etapas 5 y 6 (Clausura y Dilatación)
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 8))
    fig3.suptitle('Proceso de Detección - Etapas 5 y 6', fontsize=16, fontweight='bold')
    
    axes3[0].imshow(canny_closed, cmap='gray')
    axes3[0].set_title('5. Clausura Morfológica (Conectar Bordes)', fontsize=14, fontweight='bold')
    axes3[0].axis('off')
    
    axes3[1].imshow(canny_dilated, cmap='gray')
    axes3[1].set_title('6. Dilatación de Bordes (Engrosamiento)', fontsize=14, fontweight='bold')
    axes3[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # ETAPA 8: Detección y clasificación final de componentes
    imagen_con_detecciones = imagen_rgb.copy()
    
    # Contadores para diferentes tipos de componentes
    contador_resistencias = 0
    contador_chips = 0  
    contador_circulos_pequenos = 0
    contador_circulos_medianos = 0
    contador_circulos_grandes = 0
    
    # Listas para almacenar posiciones de etiquetas
    posiciones_resistencias = []
    posiciones_chips = []
    posiciones_circulos = []
    
    # Procesar cada componente conectada
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        if area > 3200:
            # Calcular relación de aspecto (proporción ancho/alto)
            aspect_ratio = max(w, h) / min(w, h)
            
            if aspect_ratio < 1.7:
                # Buscar componentes circulares usando transformada de Hough
                submask = imagen_rgb[y:y+h, x:x+w]
                gris_submask = cv2.cvtColor(submask, cv2.COLOR_BGR2GRAY)
                gris_submask = cv2.GaussianBlur(gris_submask, (9, 9), 0)
                circulos = cv2.HoughCircles(gris_submask, cv2.HOUGH_GRADIENT, dp=1, minDist=250, 
                                          param1=70, param2=50, minRadius=25, maxRadius=200)
                
                if circulos is not None:
                    circulos = np.round(circulos[0, :]).astype("int")
                    
                    # Definir umbrales de radio para clasificar círculos por tamaño
                    umbral_pequeno = 100
                    umbral_mediano = 150
                    
                    # Procesar cada círculo detectado
                    for circle in circulos:
                        x_c, y_c, r = circle
                        
                        # Crear máscara circular
                        mask = np.zeros(gris_submask.shape[:2], dtype=np.uint8)
                        cv2.circle(mask, (x_c, y_c), r-5, 255, -1)  # -5 para evitar píxeles del borde
                        
                        # Calcular brillo del interior del círculo
                        pixeles_circulo = gris_submask[mask == 255]
                        valor_promedio = np.mean(pixeles_circulo)
                        
                        # Verificar si el interior del círculo es suficientemente brillante (blanco)
                        umbral_brillo = 130
                        
                        if valor_promedio > umbral_brillo:
                            # Agregar el desplazamiento de la submáscara para obtener coordenadas globales
                            global_x = x + x_c
                            global_y = y + y_c
                            
                            # Dibujar círculos con diferentes colores según el tamaño
                            if r < umbral_pequeno:
                                color = (255, 0, 0)  # Rojo para círculos pequeños
                                contador_circulos_pequenos += 1
                                posiciones_circulos.append((global_x, global_y - r - 10, "Capacitor P"))
                            elif r < umbral_mediano:
                                color = (0, 255, 0)  # Verde para círculos medianos
                                contador_circulos_medianos += 1
                                posiciones_circulos.append((global_x, global_y - r - 10, "Capacitor M"))
                            else:
                                color = (0, 0, 255)  # Azul para círculos grandes
                                contador_circulos_grandes += 1
                                posiciones_circulos.append((global_x, global_y - r - 10, "Capacitor G"))
                            
                            cv2.circle(imagen_con_detecciones, (global_x, global_y), r, color, 6)
                            
            elif area > 10000:
                # Componentes muy grandes identificados como CHIPS
                color = (255, 0, 255)  # Magenta para capacitores
                cv2.rectangle(imagen_con_detecciones, (x, y), (x + w, y + h), color, 6)
                contador_chips += 1
                posiciones_chips.append((x + w//2, y - 15, "CHIP"))
            else:
                # Detectar resistencias por análisis de color
                submask = imagen_rgb[y:y+h, x:x+w]
                
                # Color objetivo en RGB (ajustable según el tipo de resistencia)
                color_objetivo = np.array([200, 160, 100], dtype=np.uint8)
                
                # Umbral de diferencia permitida (ajustable)
                umbral_color = 50
                
                # Crear máscara de píxeles que están dentro del rango de color
                limite_inferior = np.clip(color_objetivo - umbral_color, 0, 255)
                limite_superior = np.clip(color_objetivo + umbral_color, 0, 255)
                
                # Aplicar la detección de color
                mask_color = cv2.inRange(submask, limite_inferior, limite_superior)
                
                # Calcular estadísticas de coincidencia de color
                total_pixeles = submask.size // 3  # dividir entre 3 por los 3 canales RGB
                pixeles_coincidentes = np.count_nonzero(mask_color)  # contar píxeles blancos (255)
                porcentaje = (pixeles_coincidentes / total_pixeles) * 100
                
                # Si el porcentaje de coincidencia supera el umbral, es una resistencia
                if porcentaje > 23:
                    color = (0, 255, 255)  # Cian para resistencias
                    cv2.rectangle(imagen_con_detecciones, (x, y), (x + w, y + h), color, 6)
                    contador_resistencias += 1
                    posiciones_resistencias.append((x + w//2, y - 15, "RESISTENCIA"))
    
    # Agregar etiquetas de texto a la imagen
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    
    # Etiquetas para resistencias
    for pos_x, pos_y, texto in posiciones_resistencias:
        # Fondo blanco para el texto
        (text_width, text_height), baseline = cv2.getTextSize(texto, font, font_scale, font_thickness)
        cv2.rectangle(imagen_con_detecciones, 
                     (pos_x - text_width//2 - 5, pos_y - text_height - 5),
                     (pos_x + text_width//2 + 5, pos_y + 5),
                     (255, 255, 255), -1)
        # Texto negro
        cv2.putText(imagen_con_detecciones, texto, (pos_x - text_width//2, pos_y), 
                   font, font_scale, (0, 0, 0), font_thickness)
    
    # Etiquetas para capacitores
    for pos_x, pos_y, texto in posiciones_chips:
        (text_width, text_height), baseline = cv2.getTextSize(texto, font, font_scale, font_thickness)
        cv2.rectangle(imagen_con_detecciones, 
                     (pos_x - text_width//2 - 5, pos_y - text_height - 5),
                     (pos_x + text_width//2 + 5, pos_y + 5),
                     (255, 255, 255), -1)
        cv2.putText(imagen_con_detecciones, texto, (pos_x - text_width//2, pos_y), 
                   font, font_scale, (0, 0, 0), font_thickness)
    
    # Etiquetas para círculos
    for pos_x, pos_y, texto in posiciones_circulos:
        (text_width, text_height), baseline = cv2.getTextSize(texto, font, font_scale, font_thickness)
        cv2.rectangle(imagen_con_detecciones, 
                     (pos_x - text_width//2 - 5, pos_y - text_height - 5),
                     (pos_x + text_width//2 + 5, pos_y + 5),
                     (255, 255, 255), -1)
        cv2.putText(imagen_con_detecciones, texto, (pos_x - text_width//2, pos_y), 
                   font, font_scale, (0, 0, 0), font_thickness)
    
    # FIGURA 4: Etapas 7 y 8 (Componentes Conectadas y Resultado Final)
    fig4, axes4 = plt.subplots(1, 2, figsize=(16, 8))
    fig4.suptitle('Proceso de Detección - Etapas 7 y 8 (Resultado Final)', fontsize=16, fontweight='bold')
    
    axes4[0].imshow(labels_colored)
    axes4[0].set_title(f'7. Componentes Conectadas Identificadas ({num_labels-1} componentes)', fontsize=14, fontweight='bold')
    axes4[0].axis('off')
    
    axes4[1].imshow(imagen_con_detecciones)
    
    titulo_final = f'8. Resultado Final - '

    total_circulos = contador_circulos_pequenos + contador_circulos_medianos + contador_circulos_grandes

    axes4[1].set_title(titulo_final, fontsize=14, fontweight='bold')
    axes4[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Crear texto de clasificación detallada y estadísticas
    texto_clasificacion = []
    texto_clasificacion.append("═══ CLASIFICACIÓN DE COLORES EN LA DETECCIÓN ═══")
    texto_clasificacion.append("")
    
    if contador_resistencias > 0:
        texto_clasificacion.append(f"🟦 CELESTE: Resistencias detectadas ({contador_resistencias} unidades)")
    if contador_chips > 0:
        texto_clasificacion.append(f"🟣 MAGENTA: Chips detectados ({contador_chips} unidades)")
    if contador_circulos_pequenos > 0:
        texto_clasificacion.append(f"🔴 ROJO: Capacitores pequeños ({contador_circulos_pequenos} unidades)")
    if contador_circulos_medianos > 0:
        texto_clasificacion.append(f"🟢 VERDE: Capacitores medianos ({contador_circulos_medianos} unidades)")
    if contador_circulos_grandes > 0:
        texto_clasificacion.append(f"🔵 AZUL: Capacitores grandes ({contador_circulos_grandes} unidades)")
    
    texto_clasificacion.append("")
    texto_clasificacion.append("═══ RESUMEN ESTADÍSTICO ═══")
    texto_clasificacion.append(f"Total de componentes analizados: {num_labels - 1}")
    texto_clasificacion.append(f"Total de resistencias encontradas: {contador_resistencias}")
    texto_clasificacion.append(f"Total de chips: {contador_chips}")
    texto_clasificacion.append(f"Total de capacitores encontrados: {contador_circulos_grandes + contador_circulos_medianos + contador_circulos_pequenos}")
    
    # Mostrar clasificación final
    print('\n'.join(texto_clasificacion))
    
    # Imprimir resumen detallado en consola
    print("\n" + "=" * 60)
    print("           REPORTE DETALLADO DE DETECCIÓN")
    print("=" * 60)
    print(f"Imagen analizada: {ruta_imagen}")
    print(f"Componentes totales detectados: {num_labels - 1}")
    print("-" * 60)
    print("CLASIFICACIÓN POR TIPO:")
    print(f"  • Resistencias: {contador_resistencias}")
    print(f"  • Chips: {contador_chips}")
    print(f"  • Capacitores pequeños: {contador_circulos_pequenos}")
    print(f"  • Capacitores medianos: {contador_circulos_medianos}")
    print(f"  • Capacitores grandes: {contador_circulos_grandes}")
    print("=" * 60)
    
    return {
        'total_resistencias': contador_resistencias,
        'total_chips': contador_chips,
        'total_capacitores': total_circulos,
        'total_componentes': num_labels - 1
    }

# Función principal para ejecutar el programa
if __name__ == "__main__":
    # Ruta de la imagen de la placa (modificar según sea necesario)
    ruta_imagen = './placa.png'
    
    print("Iniciando análisis de la placa electrónica...")
    print("Procesando imagen, por favor espere...")
        
    resultados = detectar_resistencias_parejas(ruta_imagen)
        
    print("\nAnálisis completado")
    
        
