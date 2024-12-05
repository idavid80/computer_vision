import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Obtener fotogramas
    ret, frame = cap.read()
    # Mientras obtenga imagen
    if ret == True:
        # Convertir el fotograma a espacio de color HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cv2.imshow('frame', frame)
        cv2.imshow('BGR', np.hstack([frame[:,:,0], frame[:,:,1], frame[:,:,2]]))
        cv2.imshow('HSV', np.hstack([hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]]))

        # 0xFF debe utilizarse con máquinas de 64 bits
        # ord('s') indica que s cancela
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Finalizar la captura
cap.release()
# Eliminar cualquier ventana abierta
cv2.destroyAllWindows()
