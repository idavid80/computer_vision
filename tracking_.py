import cv2
import numpy as np

# Definir el rango de colores en formato HSV
color_bajo = np.array([45, 100, 50])     # Rango bajo de color verde en HSV
color_alto = np.array([75, 255, 255])    # Rango alto de color verde en HSV

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Leer un fotograma
    ret, frame = cap.read()
    
    if ret:
        # Convertir el fotograma a espacio de color HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Crear una máscara para el rango de colores
        mask = cv2.inRange(hsv, color_bajo, color_alto)
        
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
                    
                    # Calcular la cantidad de verde en el objeto
                    roi = frame[y:y+h, x:x+w, :]
                    cantidad_verde = roi[:,:,1].astype(float).sum() / 255
                    cv2.putText(frame, f'Verde: {cantidad_verde:.2f}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Mostrar la imagen con el objeto detectado y las etiquetas
        cv2.imshow('Frame', frame)
        
        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()

