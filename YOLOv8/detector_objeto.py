import cv2
import numpy as np
from ultralytics import YOLO

# Cargar el modelo YOLO desde el archivo yolov8n.pt
model = YOLO('yolov8n.pt') 

# Función para procesar el video de la cámara y contar persona
def count_object():
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

        object_count = 0

  # Iterar sobre las detecciones
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()  # Convertir la caja delimitadora a formato NumPy
                score = box.conf[0].cpu().numpy()  # Obtener la confianza
                label = int(box.cls[0].cpu().numpy())  # Obtener la clase

                """
                names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
                19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
                31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                78: 'hair drier', 79: 'toothbrush'}
                """
                if label == 0 and bbox[2] > 100 and bbox[3] > 100:  # numero etiqueta determina el objeto
                    object_count += 1

                    # Dibujar la caja delimitadora en el fotograma
                    xmin, ymin, xmax, ymax = map(int, bbox)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f'tv {score:.2f}', (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Mostrar el fotograma con las cajas delimitadoras y el conteo de persona
        cv2.putText(frame, f'count: {object_count}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.namedWindow("Detection",cv2.WINDOW_NORMAL)
        cv2.imshow('Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la función para contar persona desde la cámara
if __name__ == '__main__':
    count_object()
