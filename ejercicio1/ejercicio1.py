import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread('placa.png')
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

plt.imshow(imagen_rgb)
plt.title('Placa') 
plt.axis('off') 
plt.show()

# -------------------------------------------

 # Convertir a escala de grises

gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
plt.imshow(gris, cmap='gray')
plt.axis('off') 
plt.show()


# -------------------------------------------

# Filtro Gausseano para suavisar bordes

f_blur = cv2.GaussianBlur(gris,(10,10),1)
plt.imshow(f_blur, cmap='gray')
plt.axis('off')
plt.show()

# -------------------------------------------

# Canny

canny = cv2.Canny(f_blur, 50, 80)
plt.imshow(canny, cmap='gray')
plt.axis('off')
plt.show()

# -------------------------------------------

# Kernel con estructura rectangular
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))

# Aplicamos clausura (closing): dilatación seguida de erosión
canny_closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

# Mostrar resultado
plt.imshow(canny_closed, cmap='gray')
plt.axis('off')
plt.title('Canny + Clausura')
plt.show()

# -------------------------------------------

# Dilatación de las lineas

# Kernel para engrosar las líneas
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  

# Dilatación
canny_dilated = cv2.dilate(canny_closed, kernel, iterations=1)

plt.figure(figsize=(8, 6))
plt.imshow(canny_dilated, cmap='gray')
plt.title('Canny Engrosado (Dilatación)')
plt.axis('off')
plt.show()

# -------------------------------------------

# Componentes conectadas

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(canny_dilated, connectivity=8)

print(f"Número total de componentes conectadas: {num_labels - 1}")

# Visualizar todas las componentes con colores
labels_colored = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  

for i in range(num_labels):
    labels_colored[labels == i] = colors[i]

plt.figure(figsize=(12, 6))
plt.imshow(labels_colored)
plt.title(f'Todas las Componentes Conectadas ({num_labels-1} componentes)')
plt.axis('off')
plt.show()

#---------------------------------------------------------

