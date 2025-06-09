# Detección de Componentes Electrónicos y Análisis de Resistencias

Este proyecto contiene tres ejercicios para procesar imágenes de componentes electrónicos utilizando OpenCV y técnicas de visión por computadora.

## Estructura del Proyecto

```
proyecto/
├── ejercicio1.py          # Detección de componentes en placas electrónicas
├── ejercicio2_parte1.py   # Transformación de perspectiva de resistencias
├── ejercicio2_parte2.py   # Análisis de colores y cálculo de valores
├── imagenes/             # Carpeta con imágenes originales (.jpg)
├── imagenes_out/         # Carpeta de salida (transformaciones)
├── imagenes_intermedias/ # Carpeta de análisis detallado
└── placa.png            # Imagen de placa para el ejercicio 1
```

## Ejercicio 1: Detección de Componentes en Placas

**Archivo:** `ejercicio1.py`

Detecta y clasifica componentes electrónicos en imágenes de placas:
- **Resistencias** (marcadas en celeste)
- **Chips** (marcados en celeste/magenta)
- **Capacitores** pequeños, medianos y grandes (rojo, verde, azul)

### Uso
```bash
python ejercicio1.py
```

Coloca tu imagen de placa como `placa.png` en el directorio raíz.

## Ejercicio 2 Parte 1: Transformación de Perspectiva

**Archivo:** `ejercicio2_parte1.py`

Aplica transformación de perspectiva a imágenes de resistencias para enderezarlas:
- Detecta regiones azules en las resistencias
- Corrige la perspectiva automáticamente
- Guarda las imágenes transformadas

### Uso
```bash
python ejercicio2_parte1.py
```

- **Entrada:** Archivos `.jpg` en carpeta `imagenes/`
- **Salida:** Archivos `*_out.jpg` en carpeta `imagenes_out/`

## Ejercicio 2 Parte 2: Análisis de Colores de Resistencias

**Archivo:** `ejercicio2_parte2.py`

Analiza las bandas de colores en resistencias y calcula sus valores:
- Detecta automáticamente las bandas de colores
- Identifica colores: Negro, Marrón, Rojo, Naranja, Amarillo, Verde, Azul, Violeta, Gris, Blanco
- Calcula el valor de resistencia en Ω, kΩ o MΩ
- Genera imágenes paso a paso del procesamiento

### Uso
```bash
python ejercicio2_parte2.py
```

- **Entrada:** Archivos `*_a_out.jpg` en carpeta `imagenes_out/`
- **Salida:** 
  - Imágenes de procesamiento en `imagenes_intermedias/`
  - Archivo de resumen `resumen_resistencias.txt`

## Requisitos

```bash
pip install opencv-python numpy matplotlib
```

## Flujo de Trabajo Recomendado

1. **Para placas electrónicas:** Ejecutar `ejercicio1.py`
2. **Para resistencias:** 
   - Primero ejecutar `ejercicio2_parte1.py` (transformación)
   - Luego ejecutar `ejercicio2_parte2.py` (análisis de colores)

## Características Principales

- **Detección automática** de componentes y colores
- **Visualización paso a paso** del procesamiento
- **Corrección automática** de orientación
- **Cálculo preciso** de valores de resistencia
- **Generación de reportes** detallados
