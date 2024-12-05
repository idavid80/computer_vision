import cv2
import numpy as np
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# Cargar el modelo YOLO desde el archivo yolov8n.pt
model = YOLO('yolov8n.pt')

# Inicializar el filtro de Kalman
kalman = cv2.KalmanFilter(4, 2)

# Configurar la matriz de medidas
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

# Configurar la matriz de transición
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

# Configurar la matriz de covarianza del proceso
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

# Definir la función de IoU sin torch
def get_iou_np(ground_truth, pred):
    # Coordinates of the area of intersection.
    ix1 = max(ground_truth[0], pred[0])
    iy1 = max(ground_truth[1], pred[1])
    ix2 = min(ground_truth[2], pred[2])
    iy2 = min(ground_truth[3], pred[3])
    
    # Intersection height and width.
    i_height = max(iy2 - iy1 + 1, 0)
    i_width = max(ix2 - ix1 + 1, 0)
    
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

def detect_person():
    # Variable para contar los fotogramas sin detección
    frames_without_detection = 0

    cap = cv2.VideoCapture(0)  # Abrir la cámara

    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    # Diccionario para almacenar las detecciones previas con sus IDs
    previous_detections = {}
    next_id = 1

    while True:
        ret, frame = cap.read()  # Leer un fotograma de la cámara

        if not ret:
            break

        # Realizar la inferencia con YOLO en el fotograma actual
        results = model(frame)
        
        detected_objects = []
        detected = False

        # Iterar sobre las detecciones
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()  # Convertir la caja delimitadora a formato NumPy
                score = box.conf[0].cpu().numpy()  # Obtener la confianza
                label = int(box.cls[0].cpu().numpy())  # Obtener la clase

                if label == 0 and bbox[2] > 100 and bbox[3] > 100:  # Ajustar, persona es 0, según modelo
                    detected = True
                    frames_without_detection = 0  # Reiniciar el contador

                    # Guardar la detección actual
                    detected_objects.append((bbox, score))

        if not detected:
            frames_without_detection += 1

        # Asociar detecciones nuevas con las previas
        for detected_bbox, score in detected_objects:
            xmin, ymin, xmax, ymax = map(int, detected_bbox)

            # detected_array = np.array(detected_bbox, dtype=np.float32)

            #  Un tensor es una estructura de datos similar a un arreglo numpy, pero optimizada para realizar operaciones en GPU
            detected_tensor = torch.tensor(detected_bbox, dtype=torch.float32)

            best_iou = 0
            best_id = None

            # iterar sobre cada detección previa almacenada en el diccionario previous_detections.
            # clave obj_id y valores una tupla que contiene la caja delimitadora (prev_bbox) y el filtro de Kalman asociado (prev_kalman).

            for obj_id, (prev_bbox, prev_kalman) in previous_detections.items():

                print('prev_kalman', prev_kalman)
                
                # iou = get_iou_np(prev_bbox, detected_array)
                
                iou = get_iou_torch(prev_bbox, detected_tensor)
                
                if iou > best_iou:
                    best_iou = iou
                    best_id = obj_id

            if best_iou > 0.5:  # Si el IoU es mayor a 0.5, consideramos que es el mismo objeto
                obj_id = best_id
                # previous_detections[obj_id] = (detected_array, previous_detections[obj_id][1])

                previous_detections[obj_id] = (detected_tensor, previous_detections[obj_id][1])
               
            else:
                obj_id = next_id
                next_id += 1
                # previous_detections[obj_id] = (detected_array, kalman)

                previous_detections[obj_id] = (detected_tensor, kalman)                

            # Dibujar la caja delimitadora en el fotograma
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id} {score:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Aplicar el filtro de Kalman
            measurement = np.array([[np.float32((xmin + xmax) / 2)],
                                    [np.float32((ymin + ymax) / 2)]])
            previous_detections[obj_id][1].correct(measurement)

        # Predecir la posición del filtro de Kalman para todos los objetos
        for obj_id, (bbox, kalman_filter) in previous_detections.items():
            prediction = kalman_filter.predict()
            pred_x, pred_y = int(prediction[0]), int(prediction[1])

            pred_bbox = [pred_x - 50, pred_y - 50, pred_x + 50, pred_y + 50]

            # Dibujar la predicción del filtro de Kalman solo si han pasado menos de 5 fotogramas sin detección
            if detected or frames_without_detection <= 15:
                if not detected:
                    cv2.putText(frame, 'Predicción', (pred_x - 20, pred_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)
                cv2.rectangle(frame, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (255, 0, 0), 2)

        # Mostrar el fotograma con las cajas delimitadoras y el conteo de cups
        cv2.putText(frame, f'Contador: {len(previous_detections)}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.namedWindow("Persona detectada", cv2.WINDOW_NORMAL)
        cv2.imshow('Persona detectada', frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la función para contar cups desde la cámara
if __name__ == '__main__':
    detect_person()
