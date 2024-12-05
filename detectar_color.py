import cv2

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Leer un fotograma
    ret, frame = cap.read()
    
    if ret:
        # Convertir el fotograma a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular la cantidad de verde sumando los valores de los píxeles en el canal verde
        cantidad_verde = frame[:,:,1].astype(float).sum() / 255
        
        # Mostrar el fotograma y la cantidad de verde en la consola
        cv2.imshow('Frame', frame)
        print("Cantidad de verde:", cantidad_verde)
        
        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()