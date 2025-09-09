# Modelos de Clasificación de Imágenes para Dígitos y Direcciones

Este repositorio contiene dos cuadernos (Jupyter notebooks) para entrenar modelos de Redes Neuronales Convolucionales (CNN) para tareas de clasificación de imágenes usando TensorFlow y Keras.

## Descripción General

El proyecto consiste en dos scripts de entrenamiento separados:

1. `entrenar_nums.ipynb`: Entrena un modelo para reconocer dígitos escritos a mano (0-9).

2. `entrenar_direct.ipynb`: Entrena un modelo para clasificar imágenes como 'izquierda' o 'derecha'.

Ambos cuadernos utilizan una arquitectura CNN similar y un proceso de carga de datos desde directorios locales.

## Modelos

### 1. Reconocimiento de Dígitos (`entrenar_nums.ipynb`)

Este cuaderno entrena un modelo para clasificar imágenes en escala de grises de dígitos escritos a mano.

- **Conjunto de Datos (Dataset)**:

  - Se espera que esté en el directorio `assets_num/`.

  - Debe contener 10 subdirectorios, uno para cada dígito (de '0' a '9').
  
  - Las imágenes se redimensionan a 28x28 píxeles.

- **Entrenamiento**:
  - El modelo se entrena durante 100 épocas (epochs).
  - El progreso del entrenamiento se registra para su visualización en TensorBoard.
- **Salida**:
  - El modelo entrenado se guarda como `numeros2.keras`.

### 2. Clasificación de Dirección (`entrenar_direct.ipynb`)

Este cuaderno entrena un modelo para clasificar imágenes que representan 'izquierda' o 'derecha'.

- **Conjunto de Datos (Dataset)**:
  - Se espera que esté en el directorio `assets_direct/`.

  - Debe contener 2 subdirectorios: `izquierda` y `derecha`.

  - Las imágenes se redimensionan a 55x94 píxeles.

- **Entrenamiento**:

  - El modelo se entrena durante 50 épocas (epochs).

  - El progreso del entrenamiento se registra para su visualización en TensorBoard.

- **Salida**:

  - El modelo entrenado se guarda como `direccion.h5`.

## Arquitectura del Modelo

Ambos modelos comparten la siguiente arquitectura CNN:

1. Capa de Entrada (`Input`) (`(alto, ancho, 1)`)
2. `Conv2D` (32 filtros, kernel 3x3, activación ReLU)
3. `MaxPooling2D` (tamaño de agrupación 2x2)
4. `BatchNormalization`
5. `Conv2D` (64 filtros, kernel 3x3, activación ReLU)
6. `MaxPooling2D` (tamaño de agrupación 2x2)
7. `BatchNormalization`
8. `Flatten`
9. `Dense` (128 unidades, activación ReLU)
10. `Dropout` (tasa de 0.5)
11. `Dense` (capa de salida con `num_classes` unidades y activación Softmax)

## Requisitos

Para ejecutar estos cuadernos, necesitas Python y las siguientes librerías:

- `tensorflow`
- `numpy`
- `matplotlib`
- `jupyter`

Puedes instalarlas usando pip:

```bash
pip install tensorflow numpy matplotlib jupyterlab
```

## Uso

1. **Preparar los conjuntos de datos**:
    - Crea un directorio llamado `assets_num/` y coloca las imágenes de los dígitos en sus respectivas carpetas de clase (ej. `assets_num/0/`, `assets_num/1/`, etc.).
    - Crea un directorio llamado `assets_direct/` y coloca las imágenes de las direcciones en sus respectivas carpetas de clase (`assets_direct/izquierda/`, `assets_direct/derecha/`).
2. **Ejecutar los cuadernos**:
    - Inicie Jupyter Lab o Jupyter Notebook.
    - Abra y ejecute las celdas en `entrenar_nums.ipynb` para entrenar el modelo de reconocimiento de dígitos.
    - Abra y ejecute las celdas en `entrenar_direct.ipynb` para entrenar el modelo de clasificación de dirección.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
