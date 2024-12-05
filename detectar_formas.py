import cv2

# cap = cv2.VideoCapture(0)
img = cv2.imread('./images/formas.png')

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detección de bordes
canny = cv2.Canny(grey, 10,150)
# dilatar y erosionar imagen para corregir bordes no completados
canny = cv2.dilate(canny, None, iterations=1)
canny = cv2.erode(canny, None, iterations=1)

contour, _ =cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contour:
    # dibujamos el contorno
    cv2.drawContours(img, [c], 0, (0,255,0), 2)

    epsilon = 0.01*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    x, y, w, h = cv2.boundingRect(approx)
    print(len(approx))
    if len(approx) == 3:
        cv2.putText(img, "Triangulo", (x, y -5), 1, 1, (0, 255, 0), 1)
    if len(approx) == 4:
        aspect_ratio = float(w)/h
        print(aspect_ratio)
        if aspect_ratio == 1:
            cv2.putText(img, "Cuadrado", (x, y - 5), 1, 1, (0, 255, 0), 1)
        else:
            cv2.putText(img, "Rectangulo", (x, y - 5), 1, 1, (0, 255, 0), 1)
    if len(approx) == 5:
        cv2.putText(img, "Pentagono", (x, y -5), 1, 1, (0, 255, 0), 1)
    if len(approx) == 6:
        cv2.putText(img, "Hexagono", (x, y -5), 1, 1, (0, 255, 0), 1)
    if len(approx) >= 13:
        cv2.putText(img, "Circulo", (x, y -5), 1, 1, (0, 255, 0), 1)
        
    
    cv2.drawContours(img, [approx], 0, (0,255,0),2)
    cv2.imshow('Polig', img)
    cv2.waitKey(0)


cv2.imshow('Polig', img)
cv2.waitKey(0)

cv2.destroyAllWindows()

def getForm(frame):
    #image = cv2.imread(frame)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detección de bordes
    canny = cv2.Canny(grey, 10,150)

    _, contour, _ =cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contour:
        cv2.drawContours(frame, [c], 0, (0,255,0), 2)
        cv2.imshow('Polig', frame)
        cv2.waitKey(0)

"""
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

def stackt_images(scale, imgArray):

    rows = len(imgArray)
    columns = len(imgArray[0])

    rowsAvailable = isinstance(imgArray[0], list)

    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, columns):

"""