import cv2

def detectForm(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detección de bordes
    canny = cv2.Canny(grey, 10, 150)
    # Dilatar y erosionar imagen para corregir bordes no completados
    canny = cv2.dilate(canny, None, iterations=1)
    canny = cv2.erode(canny, None, iterations=1)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # Dibujamos el contorno
        cv2.drawContours(img, [c], 0, (0, 255, 0), 2)

        epsilon = 0.1 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:
            cv2.putText(img, "Triangulo", (x, y - 5), 1, 1, (0, 255, 0), 1)
        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            # Use a threshold for aspect ratio comparison
            if 0.95 <= aspect_ratio <= 1.05:
                cv2.putText(img, "Cuadrado", (x, y - 5), 1, 1, (0, 255, 0), 1)
            else:
                cv2.putText(img, "Rectangulo", (x, y - 5), 1, 1, (0, 255, 0), 1)
        elif len(approx) == 5:
            cv2.putText(img, "Pentagono", (x, y - 5), 1, 1, (0, 255, 0), 1)
        elif len(approx) == 6:
            cv2.putText(img, "Hexagono", (x, y - 5), 1, 1, (0, 255, 0), 1)
        elif len(approx) >= 13:
            cv2.putText(img, "Circulo", (x, y - 5), 1, 1, (0, 255, 0), 1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Obtener fotogramas
    ret, frame = cap.read()
    # Mientras obtenga imagen
    if ret:
        detectForm(frame)
        
        # Mostrar el fotograma procesado
        cv2.imshow('Polig', frame)

        # 0xFF debe utilizarse con máquinas de 64 bits
        # ord('q') indica que 'q' cancela
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Finalizar la captura
cap.release()
# Eliminar cualquier ventana abierta
cv2.destroyAllWindows()
