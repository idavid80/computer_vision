import cv2
import numpy as np

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    img = np.int16(image)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    return np.uint8(img)

# cap = cv2.VideoCapture(0)
img = cv2.imread('./images/manchas.jpeg')

img_adjusted = adjust_brightness_contrast(img, brightness=30, contrast=30)

grey = cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2GRAY)
# Aplicar suavizado Gaussiano
# Para poder contar los objetos o monedas vamos a aplicar el filtro Gaussiano.
# La matriz tiene que ser cuadrada e impar para poder obtener un espacio centrado (Campana de Gauss)
gaussiana = cv2.GaussianBlur(grey, (3,3), 7)
    # Detección de bordes
    # Detección de bordes con Sobel: cálculo de la primera derivada. mide las evoluciones y los cambios de una variable.
    # Supresión de píxeles fuera del borde: non-maximun permite adelgazar los bordes basándose en el gradiente.
    # Aplicar umbral por histéresis: determina un umbral que decide si el píxel forma parte del fondo o forma parte de un objeto.
canny = cv2.Canny(grey, 40,80)
# dilatar y erosionar imagen para corregir bordes no completados
canny = cv2.dilate(canny, None, iterations=1)
canny = cv2.erode(canny, None, iterations=1)
# Aplicar umbral para obtener una imagen binaria
#filtered_contours = [cnt for cnt in canny if cv2.contourArea(cnt) > 25]

(contour, _) =cv2.findContours(canny, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE)

filtered_contours = [cnt for cnt in contour if cv2.contourArea(cnt) > 600]

print("He encontrado {} objetos".format(len(contour)))
cv2.drawContours(img,contour,-1,(0,0,255), 2)

cv2.namedWindow("Encuentra manchas",cv2.WINDOW_NORMAL)

cv2.imshow('Encuentra manchas', img)
cv2.waitKey(0)
# Guardar la imagen resultante
cv2.imwrite('./images/image_victor_result.jpg', img)
# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()