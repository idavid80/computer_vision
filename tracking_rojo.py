import cv2
import numpy as np


# el rango del rojo en HSV, tenemos que contemplar tanto el inicio como el final del umbral de colores, con los valores de H.
# Elegimos el umbral de rojo en HSV
# https://omes-va.com/deteccion-de-colores/

umbral_bajo1 = (170,100,100)
umbral_alto1 = (179,255,255)
# Elegimos el segundo umbral de rojo en HSV
umbral_bajo2 = (0,100,100)
umbral_alto2 = (10,255,255)

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Leer un fotograma
    ret, frame = cap.read()
    
    if ret:
        # Convertir el fotograma a espacio de color HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Crear máscaras para los rangos de colores
        mask1 = cv2.inRange(hsv, umbral_bajo1, umbral_alto1)
        mask2 = cv2.inRange(hsv, umbral_bajo2, umbral_alto2)
        
        # Combinar las máscaras
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Dibujar los contornos y agregar etiquetas en la imagen original
        if len(contours) > 0:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filtrar contornos pequeños
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Agregar etiqueta con las coordenadas del objeto
                    cv2.putText(frame, f'({x}, {y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Calcular la cantidad de rojo en el objeto
                    roi = frame[y:y+h, x:x+w, :]
                    cantidad_rojo = roi[:,:,2].astype(float).sum() / 255
                    cv2.putText(frame, f'Rojo: {cantidad_rojo:.2f}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Mostrar la imagen con el objeto detectado y las etiquetas
        cv2.imshow('Frame', frame)
        
        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
