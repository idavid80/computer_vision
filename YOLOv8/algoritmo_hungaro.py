"""
Pasos Básicos del Algoritmo Húngaro
- Reducción de Filas: Restar el valor mínimo de cada fila a todos los elementos de la fila.
- Reducción de Columnas: Restar el valor mínimo de cada columna a todos los elementos de la columna.
- Cubrir los Ceros: Cubrir todos los ceros en la matriz con la menor cantidad de líneas horizontales y verticales.
- Reajuste de la Matriz: Si el número de líneas es igual al número de filas (o columnas), se ha encontrado una asignación óptima.
    Si no, ajustar la matriz y repetir los pasos anteriores.

 NOTA: la función linear_sum_assignment de scipy.optimize para asociar las detecciones nuevas con las previas de manera óptima basándonos en la métrica de IoU (Intersection over Union).
 El objetivo es reemplazar el método actual de asignación basado en IoU con una implementación que utilice el algoritmo húngaro para una asociación más eficiente.
"""

# Algoritmo húngaro: https://github.com/abewley/sort?tab=readme-ov-file

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# Cargar el modelo YOLO desde el archivo yolov8n.pt
model = YOLO('yolov8n.pt')

# Inicializar el filtro de Kalman
def create_kalman_filter():
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
                                       [0, 0, 0, 1]], np.float32) * 0.03
    return kalman

# Definir la función de IoU
def get_iou(ground_truth, pred):
    ix1 = max(ground_truth[0], pred[0])
    iy1 = max(ground_truth[1], pred[1])
    ix2 = min(ground_truth[2], pred[2])
    iy2 = min(ground_truth[3], pred[3])
    
    i_height = max(iy2 - iy1 + 1, 0)
    i_width = max(ix2 - ix1 + 1, 0)
    
    area_of_intersection = i_height * i_width
    
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
    
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
    
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    
    iou = area_of_intersection / area_of_union
    
    return iou



def associate_detections_to_trackers(detections, trackers, iou_threshold=0.5):
    # Si no hay trackers, devolver directamente detecciones no emparejadas y trackers vacíos
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    # Crear una matriz de IoU (Intersection over Union) con ceros
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32) 
    
    # Llenar la matriz de IoU con los valores de IoU entre cada detección y cada tracker
    # iterar sobre cada detección y cada tracker
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = get_iou(det, trk)
    
    # Aplicar el algoritmo húngaro para encontrar el emparejamiento óptimo
    # El algoritmo minimiza el costo, por lo tanto, usamos -iou_matrix para maximizar IoU

    matched_index  = linear_sum_assignment(-iou_matrix) # encuentra el emparejamiento óptimo que maximiza el IoU.
    
    # matched_index contiene los índices de detecciones y trackers emparejados

    # Crear listas para detecciones y trackers no emparejados

    unmatched_detections = [] # almacena los índices de las detecciones y trackers que no han sido emparejados
    
    for d, det in enumerate(detections):
        if d not in matched_index[0]:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_index[1]:
            unmatched_trackers.append(t)
    
    # Crear una lista para los emparejamientos validos
    matches = []

    for m in zip(matched_index[0], matched_index[1]):
        # Verificar si el IoU del emparejamiento es menor que el umbral
        if iou_matrix[m[0], m[1]] < iou_threshold:
            # Si es menor, agregar a las listas de no emparejados
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            # Si es mayor o igual, agregar a la lista de emparejamientos
            matches.append(m)
    
    # Devolver los emparejamientos, detecciones no emparejadas y trackers no emparejados
    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)


def detect_person():
    frames_without_detection = 0

    cap = cv2.VideoCapture(0)  # Abrir la cámara

    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    previous_detections = {}
    next_id = 1

    while True:
        ret, frame = cap.read()  # Leer un fotograma de la cámara

        if not ret:
            break

        results = model(frame)
        
        detected_objects = []
        detected = False

        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()  # Convertir la caja delimitadora a formato NumPy
                score = box.conf[0].cpu().numpy()  # Obtener la confianza
                label = int(box.cls[0].cpu().numpy())  # Obtener la clase

                if label == 0 and bbox[2] > 100 and bbox[3] > 100:
                    detected = True
                    frames_without_detection = 0
                    detected_objects.append((bbox, score))

        if not detected:
            frames_without_detection += 1

        # Asociar detecciones nuevas con las previas
        if len(previous_detections) > 0:
            # Obtener las cajas delimitadoras de las detecciones previas
            previous_bboxes = np.array([det[0] for det in previous_detections.values()])

            # Asociar las detecciones nuevas con las trayectorias previas usando IoU y el algoritmo húngaro
            matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(
                np.array([det[0] for det in detected_objects]), previous_bboxes)

            # Procesar los emparejamientos (matches)
            for match in matches:
                det_idx, trk_idx = match
                detected_bbox = detected_objects[det_idx][0]
                obj_id = list(previous_detections.keys())[trk_idx]
                kalman_filter = previous_detections[obj_id][1]

                # Corregir la predicción del filtro de Kalman con la medida actual
                measurement = np.array([[np.float32((detected_bbox[0] + detected_bbox[2]) / 2)],
                                        [np.float32((detected_bbox[1] + detected_bbox[3]) / 2)]])
                kalman_filter.correct(measurement)

                # Actualizar la detección previa con la nueva caja delimitadora y el filtro de Kalman corregido
                previous_detections[obj_id] = (detected_bbox, kalman_filter)

                # Dibujar la caja delimitadora y la ID en el frame
                xmin, ymin, xmax, ymax = map(int, detected_bbox)
                score = detected_objects[det_idx][1]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {obj_id} {score:.2f}', (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                
            
            # Procesar las detecciones no emparejadas (unmatched_detections)
            for det_idx in unmatched_detections:
                detected_bbox = detected_objects[det_idx][0]
                obj_id = next_id
                next_id += 1

                 # Crear un nuevo filtro de Kalman para la nueva detección
                kalman_filter = create_kalman_filter()
                previous_detections[obj_id] = (detected_bbox, kalman_filter)
                xmin, ymin, xmax, ymax = map(int, detected_bbox)
                score = detected_objects[det_idx][1]

                # Dibujar la caja delimitadora y la ID en el frame
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {obj_id} {score:.2f}', (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

           
            # Procesar los trackers no emparejados (unmatched_trackers)
            for trk_idx in unmatched_trackers:
                obj_id = list(previous_detections.keys())[trk_idx]
                kalman_filter = previous_detections[obj_id][1]

                # Predecir la nueva posición del objeto usando el filtro de Kalman
                prediction = kalman_filter.predict()
                pred_x, pred_y = int(prediction[0]), int(prediction[1])
                pred_bbox = [pred_x - 50, pred_y - 50, pred_x + 50, pred_y + 50]

                # Dibujar la predicción en el frame
                cv2.putText(frame, 'Predicción', (pred_x - 20, pred_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)
                cv2.rectangle(frame, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (255, 0, 0), 2)

        else:
            # procesar todas las detecciones actuales como nuevas
            for detected_bbox, score in detected_objects:
                obj_id = next_id
                next_id += 1

                # Crear un nuevo filtro de Kalman para cada nueva detección
                kalman_filter = create_kalman_filter()
                previous_detections[obj_id] = (detected_bbox, kalman_filter)
                xmin, ymin, xmax, ymax = map(int, detected_bbox)

                # Dibujar la caja delimitadora y la ID en el frame
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {obj_id} {score:.2f}', (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.putText(frame, f'Contador: {len(previous_detections)}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.namedWindow("Persona detectada", cv2.WINDOW_NORMAL)
        cv2.imshow('Persona detectada', frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_person()
