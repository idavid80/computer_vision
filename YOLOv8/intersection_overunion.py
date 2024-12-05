import cv2
import numpy as np
import torch
from ultralytics import YOLO

# https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/

# Cargar el modelo YOLO desde el archivo yolov8n.pt
model = YOLO('yolov8n.pt')

# Inicializar el filtro de Kalman
# objeto filtro Kalman con 4 variables de estado y 2 variables de medida utilizando cv2.KalmanFilter(4, 2)
kalman = cv2.KalmanFilter(4, 2)

# configuramos la matriz de medidas self.kalman.measurementMatrix
# una matriz de 2×4 que mapea las coordenadas x e y a nuestro vector de estado de 4 dimensiones
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)


# transitionMatrix: define cómo evolucionan nuestros vectores de estado desde el paso de tiempo t hasta t+1
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

# processNoiseCov representa la incertidumbre en nuestro modelo de movimiento y afecta a cómo el filtro Kalman predice el siguiente estado
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

# Un valor más bajo implica menos incertidumbre en la predicción del estado futuro del sistema,
# mientras que un valor más alto implica más incertidumbre y, por lo tanto, una estimación menos precisa.


# Definir la función de IoU
def get_iou_torch(ground_truth, pred):
    # Coordinates of the area of intersection.
    ix1 = torch.max(ground_truth[0], pred[0])
    iy1 = torch.max(ground_truth[1], pred[1])
    ix2 = torch.min(ground_truth[2], pred[2])
    iy2 = torch.min(ground_truth[3], pred[3])
    
    # Intersection height and width.
    i_height = torch.max(iy2 - iy1 + 1, torch.tensor(0.))
    i_width = torch.max(ix2 - ix1 + 1, torch.tensor(0.))
    
    area_of_intersection = i_height * i_width
    
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
    
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
    
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    
    iou = area_of_intersection / area_of_union
    
    return iou

def count_cup():
    # Variable para contar los fotogramas sin detección
    frames_without_detection = 0

    cap = cv2.VideoCapture(0)  # Abrir la cámara

    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    while True:
        ret, frame = cap.read()  # Leer un fotograma de la cámara

        if not ret:
            break

        # Realizar la inferencia con YOLO en el fotograma actual
        results = model(frame)
        
        mouse_count = 0
        detected = False
        # Iterar sobre las detecciones
        for result in results:

            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()  # Convertir la caja delimitadora a formato NumPy
                score = box.conf[0].cpu().numpy()  # Obtener la confianza
                label = int(box.cls[0].cpu().numpy())  # Obtener la clase
                

                if label == 0 and bbox[2] > 100 and bbox[3] > 100:  # Ajusta 0 según la clase persona en Yolov8
                    mouse_count += 1

                    detected = True
                    frames_without_detection = 0  # Reiniciar el contador

                    # Dibujar la caja delimitadora en el fotograma
                    xmin, ymin, xmax, ymax = map(int, bbox)


                    # Aplicar el filtro de Kalman
                    measurement = np.array([[np.float32((xmin + xmax) / 2)],
                                            [np.float32((ymin + ymax) / 2)]])

                    # Realizar la fase de corrección del filtro de Kalman
                    kalman.correct(measurement)

                    # Guardar ground_truth para IoU
                    ground_truth = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f'id: {len(measurement)} {score:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            if not detected:
                frames_without_detection += 1

            # Predecir la posición del filtro de Kalman
            prediction = kalman.predict()
            pred_x, pred_y = int(prediction[0]), int(prediction[1])

            if detected:
                pred_bbox = [pred_x - 50, pred_y - 50, pred_x + 50, pred_y + 50]
                get_iou = get_iou_torch(ground_truth, torch.tensor(pred_bbox, dtype=torch.float32))

                cv2.rectangle(frame, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, f'IoU: {get_iou:.2f}', (pred_bbox[0], pred_bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Dibujar la predicción del filtro de Kalman solo si han pasado menos de 5 fotogramas sin detección
            if detected or frames_without_detection <= 5:
                if not detected:
                    cv2.putText(frame, 'Predicción', (pred_x - 20, pred_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)

        # Mostrar el fotograma con las cajas delimitadoras y el conteo de cups
        cv2.putText(frame, f'Contador: {mouse_count}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.namedWindow("Taza detectada", cv2.WINDOW_NORMAL)
        cv2.imshow('Taza detectada', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la función para contar cups desde la cámara
if __name__ == '__main__':
    count_cup()

