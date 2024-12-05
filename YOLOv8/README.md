# Proyecto de Detección de Objetos en Tiempo Real con YOLOv8 y Algoritmo Húngaro
Este proyecto implementa un sistema de detección de objetos en tiempo real utilizando el modelo YOLOv8 y el algoritmo Húngaro para la asociación eficiente de las detecciones. Además, emplea un filtro de Kalman para el seguimiento de los objetos a lo largo del tiempo.

## Requisitos
- Python 3.x

## Dependencias:
- cv2 (OpenCV)
- numpy
- torch
- ultralytics
- filterpy
- scipy
- 
Además, debes tener el modelo yolov8n.pt disponible en el mismo directorio o proporcionar la ruta correcta al archivo del modelo.
Este código está diseñado para funcionar en tiempo real con la cámara, pero también puedes adaptarlo a la entrada de video desde un archivo.

### Importación de bibliotecas:
- cv2 es la biblioteca de OpenCV para trabajar con visión computacional.
- numpy se usa para operaciones matemáticas, como matrices y vectores.
- torch se usa para operaciones con tensores (como calcular IoU con PyTorch).
- YOLO de ultralytics para la detección de objetos.
- KalmanFilter de filterpy.kalman para el filtro de Kalman.

### Consideraciones Importantes:
- YOLO: Este código utiliza un modelo preentrenado de YOLOv8 (yolov8n.pt). Se debe tener este modelo disponible en el sistema.
- Filtro de Kalman: El filtro de Kalman es utilizado para predecir la ubicación futura de las personas y mejorar el seguimiento, especialmente cuando no hay detecciones en algunos fotogramas.
- IoU: La comparación de detecciones se realiza mediante IoU, y si el IoU entre una nueva detección y una detección previa es mayor que 0.5, se asume que es el mismo objeto.
- Detección de personas: Solo las detecciones que corresponden a la clase "persona" (label == 0) y con un tamaño mínimo de caja delimitadora se consideran.

#### Configuración del Filtro de Kalman:
- Se crea un filtro de Kalman con 4 variables de estado (posición y velocidad) y 2 variables de medida (posición).
- Se configura la matriz de medidas, la matriz de transición, y la matriz de covarianza del proceso.
  

## Archivos

### Archivo algoritmo_hungaro.py
El archivo se basa en la detección de personas en tiempo real utilizando el modelo preentrenado YOLOv8 de la librería Ultralytics. El sistema realiza las siguientes tareas:

#### Detección de Objetos:

Utiliza YOLOv8 para identificar objetos (en este caso, personas) en cada fotograma de la cámara en tiempo real.
Asociación de Detecciones:

Usa el algoritmo húngaro para asociar las detecciones actuales con las previas. Esto mejora la eficiencia del proceso de asignación de objetos, al optimizar la métrica de Intersection over Union (IoU).
El algoritmo Húngaro se utiliza para asociar las detecciones nuevas con las previas de manera óptima.
Seguimiento de Objetos:

Implementa un filtro de Kalman para predecir y actualizar la posición de los objetos a lo largo del tiempo.
Cada objeto detectado recibe una ID única, que se mantiene a través de los fotogramas.

#### Funcionamiento
El script captura imágenes de la cámara en tiempo real y, en cada fotograma, realiza lo siguiente:

- Detecta objetos utilizando YOLOv8.
- Asocia las detecciones actuales con los objetos previamente rastreados utilizando el algoritmo Húngaro basado en IoU.
- Actualiza la posición de los objetos en el espacio utilizando un filtro de Kalman.
- Muestra en tiempo real las detecciones y las predicciones de los objetos.

**Funciones Principales**
- create_kalman_filter: Inicializa un filtro de Kalman para el seguimiento de objetos.
- get_iou: Calcula la métrica Intersection over Union (IoU) entre dos cajas delimitadoras.
- associate_detections_to_trackers: Asocia las detecciones nuevas con los objetos previos utilizando el algoritmo Húngaro y la métrica de IoU.
- detect_person: Captura los fotogramas de la cámara, detecta objetos, realiza la asociación con los objetos previos y muestra los resultados en tiempo real.

#### Uso

Ejecuta el script:
```bash
python algoritmo_hungaro.py

```
El script comenzará a capturar imágenes desde la cámara y mostrará las detecciones en tiempo real.

Presiona q para cerrar la ventana.

#### Algoritmo Húngaro
El algoritmo Húngaro se utiliza para la asignación óptima de detecciones basadas en la métrica de IoU. Los pasos básicos incluyen:

Reducción de Filas: Restar el valor mínimo de cada fila a todos los elementos de la fila.
Reducción de Columnas: Restar el valor mínimo de cada columna a todos los elementos de la columna.
Cubrir los Ceros: Cubrir todos los ceros en la matriz con la menor cantidad de líneas horizontales y verticales.
Reajuste de la Matriz: Si el número de líneas es igual al número de filas (o columnas), se ha encontrado una asignación óptima. Si no, se ajusta la matriz y se repiten los pasos anteriores.
La función linear_sum_assignment de la librería scipy.optimize implementa esta asignación.

## Notas
Asegúrate de tener una cámara conectada y funcionando antes de ejecutar el script.
Puedes ajustar el umbral de IoU cambiando el valor de iou_threshold en la función associate_detections_to_trackers para modificar la sensibilidad del algoritmo.

### archivo detector_objeto.py

#### Descripción del código
El archivo detector_objeto.py realiza las siguientes acciones:
    - Carga el modelo YOLO: Se carga el modelo YOLO desde el archivo yolov8n.pt (debe estar presente en el directorio del proyecto o especificar su ruta correcta).
    - Captura el video desde la cámara web: Se utiliza OpenCV para capturar los fotogramas de la cámara web en tiempo real.
    - Realiza la detección de objetos: Se procesan los fotogramas con el modelo YOLO para identificar objetos en ellos.
    - Cuenta el número de objetos: El script está configurado para contar cuántas "personas" son detectados en cada fotograma. Si se desea detectar otro objeto, puedes cambiar el número de la clase en el código.
    - Dibuja cajas delimitadoras: Cuando un objeto es detectado, el script dibuja una caja alrededor del objeto y muestra su nivel de confianza.
    - Muestra el conteo en tiempo real: El número de objetos detectados se muestra en la esquina superior izquierda del video en vivo.

Salir del video: Para salir del video, presiona la tecla q.

### archivo filtro_kalman.py

#### Descripción del archivo
Este archivo utiliza el modelo YOLO (You Only Look Once) para detectar objetos en tiempo real mediante la cámara web. En este caso, el script está configurado para detectar "cups" (tazas) en un video en vivo. Además, se aplica un filtro de Kalman para realizar un seguimiento más preciso de las posiciones de los objetos detectados.

#### Características
- Detección de objetos con YOLO: Utiliza un modelo YOLO (en este caso, yolov8n.pt) para detectar objetos en tiempo real.
- Filtro de Kalman: Implementa un filtro de Kalman para estimar las posiciones futuras de los objetos detectados y mejorar la precisión del seguimiento.
- Conteo de objetos: Realiza el conteo en tiempo real de los objetos detectados (en este caso, tazas, pero se puede ajustar para otros objetos).
- Visualización: Muestra el video con las cajas delimitadoras alrededor de los objetos detectados y el conteo de las tazas, además de las predicciones del filtro de Kalman.
- 
### archivo intersection_overunion.py

#### Descripción del Código
Este archivo realiza los siguientes pasos:

1. Carga del Modelo YOLO
El código carga el modelo YOLO desde el archivo yolov8n.pt para realizar la detección de objetos. El modelo puede detectar varios objetos, y en este caso está configurado para detectar tazas.
2. Inicialización del Filtro de Kalman
El filtro de Kalman se utiliza para realizar un seguimiento más preciso de los objetos detectados. Predice la próxima posición del objeto basándose en su movimiento anterior, ayudando a suavizar las predicciones.
3. Cálculo de Intersection over Union (IoU)
La función get_iou_torch calcula la métrica Intersection over Union (IoU), que mide la superposición entre la caja delimitadora predicha y la caja delimitadora real (ground truth). Esta métrica es importante para evaluar qué tan precisas son las predicciones del modelo.
4. Captura de Video y Detección de Objetos
Se utiliza OpenCV para abrir la cámara y capturar fotogramas de video en tiempo real. Para cada fotograma, se realiza la inferencia utilizando el modelo YOLO para detectar objetos. Las cajas delimitadoras se dibujan en el fotograma, y el filtro de Kalman se aplica para predecir la posición de los objetos.
5. Predicción y Visualización con Filtro de Kalman y IoU
El filtro de Kalman predice la posición futura del objeto. Si se detecta un objeto, se calcula el valor de IoU entre la caja delimitadora predicha y la real. Este valor se dibuja sobre el fotograma junto con la caja delimitadora.
6. Conteo de Objetos
El código lleva un conteo de los objetos detectados en cada fotograma, y muestra el número total de objetos detectados en la pantalla.
7. Detener el Video
El video puede ser detenido presionando la tecla q.

#### Uso
Asegúrate de tener el archivo del modelo YOLO yolov8n.pt en el directorio correcto.

Ejecuta el script:

```bash
python intersection_overunion.py
```
El video comenzará a mostrar las detecciones de objetos (tazas), su seguimiento mediante el filtro de Kalman, el valor de IoU y el número total de 
objetos detectados.

Para salir del video, presiona la tecla q.

### archivo solucion_id.py

El archivo solucion_id.py que se encarga de realizar la detección y el seguimiento de personas con YOLO y el filtro de Kalman, utilizando técnicas de cálculo de IoU tanto en PyTorch como en NumPy:

#### Descripción del Código

1. **Funciones de Cálculo de IoU:**

- get_iou_np: Calcula el IoU (Intersection over Union) usando NumPy para dos cajas delimitadoras.
- get_iou_torch: Realiza el cálculo de IoU usando PyTorch, que permite aprovechar la capacidad de cómputo de la GPU si está disponible.

2. **Detección y Seguimiento de Personas:**
- detect_person: Es la función principal para capturar el video desde la cámara, realizar detección de personas usando YOLO y seguir a las personas usando el filtro de Kalman.

- Se inicializa una variable previous_detections para almacenar las detecciones anteriores y sus respectivos filtros de Kalman.
- Se asignan IDs a las personas detectadas, y si una detección tiene un alto IoU con una detección previa, se considera que es el mismo objeto.
- Si una persona es detectada, se aplica el filtro de Kalman para predecir su posición futura y mostrarla en el video.
- Se dibujan las cajas delimitadoras y las predicciones de los objetos con su ID y confianza en el fotograma.

3. **Interacción con la Cámara:**

- Se abre la cámara con cv2.VideoCapture(0) y se leen los fotogramas uno por uno.
- Se realiza la detección de personas en cada fotograma y se dibujan las cajas y las predicciones en el mismo.
- Se puede salir del bucle de captura presionando la tecla 'q'.

#### Resultado:

- Se muestra el fotograma con las cajas delimitadoras, el contador de personas y las predicciones de los objetos detectados.

