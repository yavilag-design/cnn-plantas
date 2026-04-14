# act6_optimiza.ipynb - Documentación

## Descripción General

Notebook de Jupyter para entrenamiento de una CNN (Red Neuronal Convolucional) basada en MobileNetV2 para clasificación de enfermedades en plantas utilizando el dataset **PlantVillage**.

## Estructura del Notebook

### 1. Imports (Celda 1)
Bibliotecas utilizadas:
- `os`, `time`, `json`, `datetime`: Utilidades del sistema
- `kagglehub`: Descarga de datasets desde Kaggle
- `cv2` (OpenCV): Procesamiento de imágenes
- `numpy`: Operaciones numéricas
- `matplotlib.pyplot`: Visualización
- `tensorflow.keras`: Redes neuronales

### 2. Carga del Dataset (Celda 2)
- Descarga el dataset `abdallahalidev/plantvillage-dataset` desde Kaggle
- Ruta: `plantvillage dataset/color`
-Dataset: ~43,456 imágenes de entrenamiento, ~10,849 de validación
- 38 clases de enfermedades de plantas

### 3. Configuración (Celda 3)
```python
TARGET_SIZE = (224, 224)    # Tamaño de imágenes de entrada
BATCH_SIZE = 32          # Tamaño de lote
KERNEL_SIZE = (5, 5)      # Tamaño del kernel Gaussiano
EPOCHS = 10              # Número de épocas
```

### 4. Preprocesamiento Personalizado (Celda 4)
Función `my_preprocessing_func()`:
- Aplica filtro Gaussiano con kernel 5x5
- Reduce ruido en las imágenes

### 5. Generadores de Datos (Celdas 5-6)
`ImageDataGenerator` con:
- `rescale=1./255`: Normalización de píxeles
- `preprocessing_function`: Filtro Gaussiano
- `validation_split=0.2`: 80% entrenamiento, 20% validación

Generadores:
- `train_generator`: 43,456 imágenes (shuffle=True)
- `val_generator`: 10,849 imágenes (shuffle=False)

### 6. Verificación Visual (Celda 7)
- Verifica forma del batch: (32, 224, 224, 3)
- Muestra ejemplo de imagen procesada

### 7. Modelo Base (Celda 8-9)
- **MobileNetV2** pre-entrenado con ImageNet
- `include_top=False`: Sin capas fully-connected
- `weights='imagenet'`: Pesos pre-entrenados
- `input_shape=(224, 224, 3)`

### 8. Arquitectura del Modelo (Celdas 10-11)
```
MobileNetV2 (base) ──┬──> GlobalAveragePooling2D ──┬──> Dropout(0.3)
                     │                         ├──> Dense(256, relu)
                     │                         ├──> BatchNormalization
                     │                         ├──> Dropout(0.3)
                     │                         ├──> Dense(128, relu)
                     │                         └──> Dense(38, softmax)
```
- Parámetros totales: 2,624,742 (~10 MB)
- Parámetros entrenables: 366,246 (~1.4 MB)
- Parámetros no entrenables: 2,258,496 (~8.62 MB)

### 9. Compilación (Celda 12)
```python
optimizer = Adam(learning_rate=0.001)
loss = categorical_crossentropy
metrics: accuracy, precision, recall, AUC, top3_accuracy
```

### 10. Callbacks (Celda 13)
- `EarlyStopping`: patience=5, restaurar mejores pesos
- `ReduceLROnPlateau`: patience=3, factor=0.5
- `ModelCheckpoint`: guardar mejor modelo en `best_model.keras`

### 11. Entrenamiento (Celda 14)
- 10 épocas
- Mejor val_accuracy: ~94.86% (época 7)
- Tiempo total: ~24,400 segundos (~6.7 horas)

### 12. Métricas Adicionales (Celdas 15-21)
- **Classification Report**: Precisión, recall, F1 por clase
- **Matriz de Confusión**: Visualización normalizada
- **Gráficos**:
  - Accuracy vs Loss
  - Precision vs Recall
  - AUC-ROC

### 13. Exportación (Celdas 22-24)
- **Modelo**: `modelo_exportado/modelo_plantas.keras`
- **Clases**: `modelo_exportado/clases.json`
- ** Métricas**: `modelo_exportado/metricas_entrenamiento.json`

## Resultados del Entrenamiento

| Métrica | Entrenamiento | Validación |
|---------|-------------|-----------|
| Accuracy | 94.76% | 94.86% |
| Loss | 0.2224 | 0.1504 |
| Precision | 95.82% | 95.82% |
| Recall | 93.90% | 93.90% |
| AUC | 99.87% | 99.87% |
| Top3 Accuracy | 99.46% | 99.46% |

## Requisitos

- Python 3.14+
- tensorflow (nightly para Python 3.14)
- scikit-learn
- opencv-python-headless
- matplotlib
- kagglehub
- numpy

## Uso

1. Ejecutar celdas en orden
2. El modelo se guarda automáticamente en `modelo_exportado/`
3. Gráficos se guardan como .png en el directorio de trabajo