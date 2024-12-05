import cv2
import numpy as np
import math

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    img = np.int16(image)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    return np.uint8(img)

def obtener_manchas(img, mask):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(grey, (3, 3), 0)
    grey = cv2.medianBlur(gaussian, 5)
    img_adjusted = adjust_brightness_contrast(grey, brightness=30, contrast=30)
    # Normalizar la iluminación y aplicar umbralización adaptativa:
    umbral = cv2.adaptiveThreshold(img_adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    canny = cv2.Canny(umbral, 30, 80)
    canny = cv2.dilate(canny, None, iterations=1)
    canny = cv2.erode(canny, None, iterations=1)
    
    # Aplicar la máscara
    masked_canny = cv2.bitwise_and(canny, canny, mask=mask)

    (contour, _) = cv2.findContours(masked_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_filtrados = [cnt for cnt in contour if cv2.contourArea(cnt) > 50]
    print("He encontrado {} objetos".format(len(contornos_filtrados)))
    cv2.drawContours(img, contour, -1, (0, 0, 255), 2)
    cv2.namedWindow("Resultado",cv2.WINDOW_NORMAL)
    cv2.imshow('Resultado', img)
 #   cv2.imwrite('./images/match-bacteria.jpg', img)

# Variables para almacenar las coordenadas de los puntos
points = []
circle_mask = None

# Función de callback para manejar eventos de mouse
def mouse_callback(event, x, y, flags, param):
    global points, circle_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Image', img)
        if len(points) == 3:
            circle_mask = calculate_circle_area()
    elif event == cv2.EVENT_RBUTTONDOWN:
        if circle_mask is not None:
            obtener_manchas(img.copy(), circle_mask)
            

# Función para calcular el centro y el radio del círculo a partir de tres puntos
def calculate_circle_area():
    global points
    A, B, C = points
    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    if D == 0:
        print("Los puntos no forman un círculo")
        return None
    Ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2) * (C[1] - A[1]) + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D
    Uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2) * (A[0] - C[0]) + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D

    r = math.sqrt((A[0] - Ux)**2 + (A[1] - Uy)**2)
    area = math.pi * r**2

    cv2.circle(img, (int(Ux), int(Uy)), int(r), (255, 0, 0), 2)
    cv2.circle(img, (int(Ux), int(Uy)), 5, (0, 0, 255), -1)
    cv2.imshow('Image', img)

    print(f"Centro del círculo: ({Ux}, {Uy})")
    print(f"Radio del círculo: {r}")
    print(f"Área del círculo: {area}")

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(Ux), int(Uy)), int(r), (255), thickness=cv2.FILLED)
    return mask

# Cargar la imagen
img = cv2.imread('./images/image_victor.jpg')
if img is None:
    print("Error: No se pudo cargar la imagen")
else:
    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mouse_callback)


    cv2.waitKey(0)

    cv2.destroyAllWindows()
