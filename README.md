# Computer vision
Proyecto básico sobre computer vision utilizando OpenCV.
Este proyecto incluye una carpeta llamada ***YOLOv8***, que contiene cinco archivos relacionados con un algoritmo de detección de objetos en tiempo real. Este algoritmo está implementado utilizando la librería Ultralytics.

## Requisitos

Antes de ejecutar este proyecto, asegúrate de tener instalados los siguientes componentes:

- Python 3.x
- OpenCV (`cv2`)

Puedes instalar OpenCV ejecutando:

```bash
pip install opencv-python
```
Instala las dependencias necesarias con:
- Numpy (`numpy`)
```bash
pip install opencv-python-headless numpy
```

Para instalar todas las dependencias de este proyecto puedes ustilizar el archivo requirements.txt, el cual incluye las dependencias de los archivos que contiene la carpera ***YOLOv8***:
```bash
pip install -r requirements.txt
```
## Cómo Ejecutar
Clona este repositorio en tu máquina local:

```bash
git clone https://github.com/idavid80/computer_vision.git
cd computer_vision
```

Ejecuta el script en tu terminal:
```bash
python nombre_archivo.py
```

## Archivos

### archivo detectar_color.py
Este proyecto utiliza **OpenCV** para capturar video desde la cámara y analizar la cantidad de verde en cada fotograma. Es ideal para quienes buscan explorar procesamiento de imágenes y computación en tiempo real.

#### Características

- Captura video en tiempo real desde la cámara.
- Convierte cada fotograma a escala de grises.
- Calcula la cantidad de verde sumando los valores del canal verde en cada fotograma.
- Muestra el video en tiempo real con la cantidad de verde registrada en la consola.
- Finaliza el programa al presionar la tecla **q**.

Permite el acceso a la cámara si es necesario.
Para salir, presiona la tecla q.

#### Funcionamiento del Código
1. El script utiliza cv2.VideoCapture(0) para acceder a la cámara principal.
2. Lee cada fotograma y calcula la cantidad de verde mediante el canal verde de la imagen en formato BGR.
3. Muestra el video en tiempo real con una ventana emergente y registra los datos en la consola.
4. Libera los recursos al finalizar.

#### Resultados
En la consola, verás algo como esto:

```yaml
Cantidad de verde: 52345.67
Cantidad de verde: 53210.42
Cantidad de verde: 54500.21
```
En paralelo, una ventana mostrará el video en tiempo real.

#### Notas
Si tu cámara no se inicia, asegúrate de que no esté en uso por otra aplicación.
La cantidad de verde se mide como la suma de los valores de los píxeles en el canal verde, normalizada por 255.

### archivo area_roi.py

Este archivo utiliza **OpenCV** para analizar imágenes, detectar manchas dentro de un área definida manualmente, y calcular propiedades geométricas como el área de un círculo. Es útil para tareas de procesamiento de imágenes relacionadas con inspecciones visuales y análisis en tiempo real.

#### Funcionalidades

- **Ajuste de brillo y contraste**: Mejora la calidad de la imagen para facilitar la detección de objetos.
- **Detección de manchas**: Identifica manchas o imperfecciones dentro de un área específica.
- **Definición de área mediante clics**: El usuario puede seleccionar tres puntos para definir un círculo donde se realizará el análisis.
- **Calculo de propiedades geométricas**: Determina el centro, el radio y el área del círculo definido.
- **Interfaz interactiva**: Usa eventos de mouse para seleccionar áreas y mostrar resultados en tiempo real.


#### Funcionamiento del Código

Sigue los pasos interactivos:
1. Selecciona tres puntos con clic izquierdo para definir el área de análisis (círculo).
2. Usa clic derecho para iniciar la detección de manchas dentro del área seleccionada.
3. Cierra las ventanas cuando termines de analizar la imagen.

#### Ejemplo de Funcionamiento
**Flujo del Programa**
1. Se carga una imagen (image_ejemplo.jpg).
2. Se seleccionan tres puntos con el mouse para definir un círculo.
3. El programa calcula:
4. Centro del círculo.
5. Radio.
6. Área.
7. Se analiza el área delimitada y se detectan manchas.
8. Los resultados se muestran en tiempo real en una ventana.
   
#### Resultados
- **En consola:**

Centro del círculo: (123.45, 678.90)
Radio del círculo: 50.34
Área del círculo: 7957.23
He encontrado 5 objetos

- **En la ventana de visualización:**
Se dibujará el círculo definido.
Las manchas detectadas estarán resaltadas.

#### Notas
Si la imagen no se carga correctamente, verifica la ruta especificada en el script.
Asegúrate de que los puntos seleccionados formen un círculo válido; el programa no puede procesar puntos colineales.
Puedes modificar el tamaño y las propiedades de las máscaras y filtros en el script para adaptarlos a tus necesidades.

### archivo detectar_formas.py

Este proyecto utiliza **OpenCV** para detectar, clasificar y etiquetar figuras geométricas presentes en una imagen. Es ideal para aplicaciones de visión por computadora donde se necesite identificar formas básicas como triángulos, rectángulos, cuadrados, pentágonos, hexágonos y círculos.

#### Características

- **Detección de bordes**: Utiliza el algoritmo de Canny para encontrar contornos en una imagen.
- **Clasificación de formas geométricas**:
  - Triángulos
  - Cuadrados
  - Rectángulos
  - Pentágonos
  - Hexágonos
  - Círculos
- **Visualización interactiva**: Las figuras detectadas se muestran con etiquetas sobre la imagen procesada.

La ventana mostrará las formas detectadas con etiquetas indicando el tipo de figura geométrica.

#### Funcionamiento del Código
1. Lectura de la imagen: Se carga la imagen especificada.
2. Conversión a escala de grises: Para simplificar la detección de bordes.
3. Detección de bordes: Utiliza el algoritmo de Canny con ajustes de dilatación y erosión para perfeccionar los contornos.

**Clasificación de figuras:**
Calcula el número de lados aproximados para determinar la figura geométrica.
Usa relaciones de aspecto para diferenciar entre cuadrados y rectángulos.
Etiquetado y visualización: Se dibujan las figuras detectadas y se añaden etiquetas en la imagen final.

#### Resultados
- **Entrada:**
Una imagen con figuras geométricas como triángulos, cuadrados, rectángulos, y círculos.

- **Salida:**
Figuras detectadas resaltadas con contornos de color verde.
Etiquetas indicando el tipo de figura geométrica sobre cada forma.

- **Consola:**
3 -> Triángulo
4 -> Rectángulo
5 -> Pentágono
6 -> Hexágono
13+ -> Círculo

- **Ventana:**
Se mostrará la imagen procesada con los contornos dibujados y las etiquetas.

#### Notas
Las figuras deben estar claramente delimitadas para una detección precisa.
Si necesitas analizar otra imagen, ajusta la ruta de entrada en el script (formas.jpg).
Puedes modificar los valores de umbral para el detector de bordes de Canny para adaptarlo a tus imágenes.
Extensiones Futuras
Permitir la detección de formas en video en tiempo real usando la cámara.
Agregar soporte para figuras con más lados.
Exportar resultados a un archivo de salida.

### archivo detectar_manchas.py

Este archivo utiliza **OpenCV** para detectar manchas u objetos en una imagen. Mediante el ajuste de brillo y contraste, y técnicas de procesamiento de imágenes como la detección de bordes y contornos, el programa identifica y resalta áreas relevantes en la imagen.

#### Características

- **Ajuste de brillo y contraste**: Mejora la calidad visual de la imagen para facilitar el procesamiento.
- **Detección de bordes**: Usa el algoritmo de Canny para identificar contornos en la imagen.
- **Filtrado de contornos**: Filtra los objetos detectados según su área para evitar el ruido.
- **Visualización interactiva**: Muestra las manchas detectadas directamente en la imagen procesada.

La ventana mostrará las manchas detectadas en la imagen.

#### Flujo del Código
1. Ajuste de brillo y contraste: Mejora los detalles visuales aplicando transformaciones sobre los valores de píxeles.
2. Conversión a escala de grises: Simplifica el análisis al trabajar en una sola escala de color.
3. Filtrado Gaussiano: Suaviza la imagen para reducir el ruido.
4. Detección de bordes con Canny:
5. Detecta cambios significativos en intensidad para identificar contornos.
6. Filtrado de contornos: Filtra objetos pequeños mediante un umbral de área mínima.
7. Dibujo de contornos: Resalta las manchas detectadas sobre la imagen original.

#### Resultados
- **Entrada:**
Una imagen cargada desde el archivo ./images/image_victor.jpg.

- **Salida:**
Una ventana interactiva muestra la imagen con las manchas detectadas resaltadas en color rojo.
Se guarda una imagen con los contornos detectados en ./images/image_victor_result.jpg.
- **Consola:**
Muestra el número total de objetos detectados, por ejemplo:
> He encontrado 8 objetos

**Ejemplo de Uso**
1. Coloca tu imagen en la carpeta images.
2. Ejecuta el script y analiza los resultados en tiempo real.
3. Verifica la imagen procesada guardada automáticamente.

#### Notas
El script está optimizado para detectar objetos con un área mínima de 600 píxeles. Puedes modificar este valor en el código según tus necesidades.
Ajusta los parámetros del detector de bordes de Canny (valores de umbral) para adaptarlo a diferentes tipos de imágenes.

#### Extensiones Futuras
- Implementar análisis en video en tiempo real.
- Incorporar opciones de análisis para diferentes tipos de manchas u objetos.
- Añadir soporte para salida en otros formatos, como datos estadísticos.

### archivo tracking_.py

Este archivo utiliza **OpenCV** para detectar y realizar un seguimiento en tiempo real de objetos con un color específico en un video capturado desde la cámara. El color objetivo se define en el espacio de color **HSV**, y se muestra información relevante como la ubicación del objeto y la cantidad de verde detectada.

#### Características

- **Detección de color**: Identifica objetos dentro de un rango de colores definido en el espacio HSV.
- **Seguimiento en tiempo real**: Rastrea el movimiento del objeto en la ventana del video.
- **Información adicional**:
  - Coordenadas del objeto detectado.
  - Cantidad de color verde en la región de interés.
- **Filtrado de ruido**: Ignora contornos pequeños para evitar detecciones falsas.

#### Flujo del Código
1. Definición del rango de colores:
2. Los colores objetivo se especifican en el espacio HSV usando un rango bajo y alto.
3. Captura de video:
4. Lee los fotogramas de la cámara en tiempo real.
5. Conversión a HSV:
6. Convierte cada fotograma de BGR (espacio de color predeterminado) a HSV para facilitar la detección por color.
7. Generación de máscara:
8. Crea una máscara binaria donde los píxeles dentro del rango de color objetivo se resaltan.
9. Detección de contornos:
10. Encuentra los bordes de las regiones resaltadas por la máscara.
11. Filtrado de contornos:
12. Ignora los contornos pequeños (área < 100 píxeles) para reducir el ruido.
13. Dibujo y etiquetado:
14. Dibuja un rectángulo alrededor de los objetos detectados y muestra:
15. Las coordenadas (x, y) de la esquina superior izquierda.
16. La cantidad de color verde en la región detectada.

#### Resultados
- **Ventana de salida:**

Muestra el video en tiempo real con los objetos detectados resaltados en verde.
Incluye etiquetas con información útil como las coordenadas y el porcentaje de color verde.

- **Consola:**
No hay salida en consola, toda la información se muestra directamente en la ventana.

- Presiona la tecla q para salir del programa.

#### Ejemplo
- **Entrada:**
Video en tiempo real desde la cámara.

- **Salida:**
Una ventana interactiva que muestra:

    - Objetos detectados resaltados con rectángulos verdes.
    - Coordenadas de cada objeto detectado.
    - Porcentaje de verde en cada región detectada.

- **Personalización:**
    - Rango de colores: Ajusta los valores de color_bajo y color_alto en el espacio HSV para detectar diferentes colores.
    - Tamaño mínimo de contornos: Modifica el valor de area > 100 para cambiar el umbral del área mínima de los contornos detectados.

#### Notas
La detección puede variar según las condiciones de iluminación. Asegúrate de realizar la captura en un ambiente con luz uniforme.
Este script utiliza la cámara predeterminada del sistema. Para cambiarla, reemplaza cv2.VideoCapture(0) con el índice o ruta del dispositivo deseado.

#### Extensiones Futuras
- Soporte para múltiples colores.
- Análisis de movimiento basado en el seguimiento del centroide del objeto.
- Integración con algoritmos de predicción para seguimiento avanzado.


### archivo tracking_rojo.py
Este archivo utiliza **OpenCV** para detectar y rastrear objetos de color rojo en tiempo real desde una cámara. La detección de color se realiza en el espacio de color **HSV**, permitiendo identificar objetos rojos incluso en condiciones de iluminación variable.

#### Características

- **Detección de color rojo**: Usa dos rangos en HSV para abarcar todos los tonos de rojo.
- **Seguimiento en tiempo real**: Identifica y rastrea objetos rojos en video en tiempo real.
- **Información adicional**:
  - Coordenadas del objeto detectado.
  - Cantidad de rojo presente en el área detectada.
- **Filtrado de ruido**: Ignora objetos pequeños para reducir falsas detecciones.

#### Funcionamiento del Código

1. Se abrirá una ventana que mostrará el video en tiempo real con los objetos detectados resaltados.
2. Definición de los umbrales de color:
3. Se utilizan dos rangos de valores en el espacio HSV:
4. Rango bajo 1: (170, 100, 100) a (179, 255, 255) para tonos más intensos.
5. Rango bajo 2: (0, 100, 100) a (10, 255, 255) para tonos más claros.
6. Captura de video:
    - Lee fotogramas en tiempo real desde la cámara.
7. Conversión a HSV:
    - Convierte cada fotograma al espacio de color HSV para detectar el color rojo de forma más precisa.
8.  Máscaras de color:
    - Genera dos máscaras (una para cada rango de rojo) y las combina para obtener la máscara final.
9.  Detección de contornos:
    - Encuentra bordes en la máscara para delimitar los objetos detectados.
    - Filtrado de contornos:
    - Descarta contornos con área menor a 100 píxeles para evitar ruido.
10. Dibujo y etiquetado:
    - Dibuja un rectángulo alrededor de los objetos detectados y muestra:
    - Coordenadas de la esquina superior izquierda del objeto.
    - Porcentaje de color rojo en la región detectada.
11. Presiona la tecla q para salir del programa.
    
#### Resultados
- **Ventana de salida:**

Muestra el video en tiempo real con objetos rojos resaltados.
Incluye etiquetas con las coordenadas y la cantidad de color rojo en cada región detectada.
- **Consola:**
No hay salida en consola; toda la información se presenta visualmente en la ventana.

#### Ejemplo
- **Entrada:**
Video en tiempo real desde la cámara.

- **Salida:**
Una ventana interactiva que muestra:

    - Objetos detectados resaltados con rectángulos verdes.
    - Coordenadas de cada objeto detectado.
    - Porcentaje de color rojo en cada región detectada.
- **Personalización**
    - Rangos de color rojo: Ajusta los valores de umbral_bajo1, umbral_alto1, umbral_bajo2, y umbral_alto2 para afinar la detección según el entorno.
    - Tamaño mínimo de contornos: Cambia el valor de area > 100 para ajustar el umbral de detección de objetos pequeños.
#### Notas
La efectividad de la detección puede verse afectada por la iluminación. Para mejores resultados, asegúrate de tener una iluminación uniforme.
Este script usa la cámara predeterminada del sistema. Para cambiarla, modifica cv2.VideoCapture(0) con el índice o ruta del dispositivo deseado.

#### Extensiones Futuras
- Implementar detección de múltiples colores simultáneamente.
- Añadir análisis de movimiento para rastrear trayectorias de objetos.
- Soporte para grabar el video con las detecciones resaltadas.

### archivo video_detector_formas.py
Este archivo utiliza **OpenCV** para detectar formas geométricas en tiempo real a partir de un flujo de video capturado por una cámara. Reconoce triángulos, cuadrados, rectángulos, pentágonos, hexágonos y círculos con base en el análisis de contornos.

#### Características

- **Detección de formas geométricas**:
  - Triángulos
  - Cuadrados
  - Rectángulos
  - Pentágonos
  - Hexágonos
  - Círculos
- **Análisis de bordes**:
  - Usa el operador Canny para detectar bordes y contornos en cada cuadro.
- **Rotulación de formas**:
  - Las formas detectadas se etiquetan directamente en el flujo de video.
- **Filtrado de bordes incompletos**:
  - Aplica dilatación y erosión para mejorar la calidad de los contornos detectados.

#### Funcionamiento del Código

**1. Procesamiento de imágenes:**
  - Convierte cada cuadro del video a escala de grises.
  - Detecta bordes usando el operador Canny.
  - Aplica dilatación y erosión para cerrar bordes incompletos.
**2. Detección de contornos:**
  - Encuentra contornos en la imagen procesada.
  - Aproxima cada contorno para determinar el número de vértices y el tipo de forma.
**3. Clasificación de formas:**
  - Basado en el número de vértices:
        - 3 vértices: Triángulo
        - 4 vértices: Cuadrado o rectángulo (según la relación de aspecto)
        - 5 vértices: Pentágono
        - 6 vértices: Hexágono
        - 13+ vértices: Círculo (forma curva aproximada)
    Para formas con 4 vértices, la relación de aspecto diferencia entre cuadrados y rectángulos.

**4. Rotulación y visualización:**
    - Dibuja los contornos detectados en verde.
    - Añade etiquetas sobre cada forma indicando su tipo.

#### Resultados
**Ventana de salida:**
Muestra el video con las formas detectadas resaltadas y etiquetadas.

#### Ejemplo
- **Entrada:**
Video en tiempo real desde la cámara.

- **Salida:**
Una ventana interactiva con:
    - Contornos de las formas detectadas en verde.
    -Etiquetas indicando el tipo de forma (e.g., "Triángulo", "Círculo").

- **Personalización**
    -Umbrales de Canny: Ajusta los valores de cv2.Canny(grey, 10, 150) para mejorar la detección de bordes según el nivel de detalle deseado.
    -Tamaño del kernel de dilatación/erosión: Modifica iterations=1 para ajustar el refinamiento de bordes.
    -Tolerancia para aproximación de contornos:
    -Cambia epsilon = 0.1 * cv2.arcLength(c, True) para ajustar la sensibilidad en la detección de vértices.

#### Notas
Este script usa la cámara predeterminada del sistema. Para cambiarla, edita cv2.VideoCapture(0) con el índice o ruta del dispositivo deseado.
Las detecciones pueden verse afectadas por condiciones de iluminación. Asegúrate de operar en un ambiente bien iluminado para mejores resultados.

#### Extensiones Futuras
- Implementar detección de colores además de formas.
- Añadir soporte para grabar el video con las formas detectadas resaltadas.
- Mejorar la detección de formas curvas o irregulares.

### archivo video_rgb.py
Este archivo utiliza **OpenCV** para capturar video en tiempo real y mostrar las intensidades de los canales **Rojo (R)**, **Verde (G)** y **Azul (B)** de cada fotograma. Es una herramienta útil para analizar los colores presentes en una imagen.

#### Características

- **Visualización en tiempo real**:
  - Muestra el video original capturado por la cámara.
  - Descompone y muestra los canales de color **R**, **G** y **B** de manera individual.
- **Interfaz interactiva**:
  - Visualización de los cuatro videos (original + tres canales de color) en ventanas separadas.
  
#### Funcionamiento del Código
1. Captura de video:
    - Usa cv2.VideoCapture(0) para capturar el flujo de video desde la cámara predeterminada.
2. Separación de canales:
    - En cada cuadro, el video se descompone en tres canales de color: Rojo (R), Verde (G) y Azul (B).
3. Visualización de canales:
    - Los canales se presentan como imágenes individuales en escala de grises para resaltar su intensidad:
    - Canal Azul: cv2.split(frame)[0]
    - Canal Verde: cv2.split(frame)[1]
    - Canal Rojo: cv2.split(frame)[2]
4. Visualización interactiva:
    - Se muestran cuatro ventanas:
    - Original: La imagen capturada tal cual.
    - Azul: Intensidad del canal azul.
    - Verde: Intensidad del canal verde.
    - Rojo: Intensidad del canal rojo.
4. Finalización del programa:
    - Presiona la tecla q para cerrar todas las ventanas y liberar los recursos.

#### Resultados
**Ventanas de salida:**
- Video Original: La imagen capturada desde la cámara.
- Canales de color (R, G, B): Muestra la intensidad de cada canal en escala de grises.

#### Ejemplo
- **Entrada:**
Video en tiempo real desde la cámara.

- **Salida:**
Cuatro ventanas mostrando:
    - Video original.
    - Canal Azul (B).
    - Canal Verde (G).
    - Canal Rojo (R).
- **Personalización**
Resolución de video:
    - Cambia la resolución predeterminada editando las propiedades de captura:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```
Fuente de video:
    - Cambia la fuente de video reemplazando cv2.VideoCapture(0) con el índice o ruta del dispositivo deseado.

#### Notas
Este script usa la cámara predeterminada del sistema. Si tienes varias cámaras, cambia el índice en cv2.VideoCapture(0) para seleccionar una diferente.
La calidad de la visualización depende de las condiciones de iluminación. Asegúrate de operar en un ambiente bien iluminado para mejores resultados.

#### Extensiones Futuras
- Agregar histogramas para analizar la distribución de colores en cada canal.
- Implementar la opción de guardar los canales individuales como imágenes.
- Añadir un selector interactivo para resaltar regiones específicas de la imagen.

## Recursos

- Para el detector de formas se utilizó el tutorial de [omes-va.com](https://omes-va.com/detectando-figuras-geometricas-con-opencv-python/)
