import cv2
import numpy as np
from ultralytics import YOLO

# Cargar el modelo YOLO desde el archivo yolov8n.pt
model = YOLO('yolov8n.pt')

# Inicializar el filtro de Kalman
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.01

# Un valor más bajo implica menos incertidumbre en la predicción del estado futuro del sistema,
# mientras que un valor más alto implica más incertidumbre y, por lo tanto, una estimación menos precisa.

def count_cup():
    cap = cv2.VideoCapture(0)  # Abrir la cámara

    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    while True:
        ret, frame = cap.read()  # Leer un fotograma de la cámara

        if not ret:
            break

        # Realizar la inferencia con YOLO en el fotograma actual
        results = model.track(frame) 

        mouse_count = 0

        # Iterar sobre las detecciones
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()  # Convertir la caja delimitadora a formato NumPy
                score = box.conf[0].cpu().numpy()  # Obtener la confianza
                label = int(box.cls[0].cpu().numpy())  # Obtener la clase

                if label == 0 and bbox[2] > 100 and bbox[3] > 100:  # Ajusta 41 según la clase del cup en tu modelo
                    mouse_count += 1

                    # Dibujar la caja delimitadora en el fotograma
                    xmin, ymin, xmax, ymax = map(int, bbox)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f'cup {score:.2f}', (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    # Aplicar el filtro de Kalman
                    measurement = np.array([[np.float32((xmin + xmax) / 2)],
                                            [np.float32((ymin + ymax) / 2)]])
                    kalman.correct(measurement)
                    prediction = kalman.predict()
                    pred_x, pred_y = int(prediction[0]), int(prediction[1])
                    
                    # Dibujar la predicción del filtro de Kalman
                    cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)

        # Mostrar el fotograma con las cajas delimitadoras y el conteo de cups
        cv2.putText(frame, f'cup count: {mouse_count}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.namedWindow("cup Detection", cv2.WINDOW_NORMAL)
        cv2.imshow('cup Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la función para contar cups desde la cámara
if __name__ == '__main__':
    count_cup()
