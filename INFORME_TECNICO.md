# INFORME TÉCNICO

## Clasificación de Enfermedades en Plantas mediante Visión por Computador y Deep Learning

**Visión por Computador - Actividad 6: Optimización y Resultados**

---

## Portada Institucional

| Campo | Información |
|-------|-------------|
| **Institución** | Universidad |
| **Curso** | Visión por Computador |
| **Unidad** | Aprendizaje Basado en Proyectos - Fase Final |
| **Actividad** | Actividad 6: Informe Técnico y Presentación de Resultados |
| **Fecha** | 14 de abril de 2026 |

---

## Resumen Ejecutivo

Este informe técnico presenta el desarrollo y evaluación de un modelo de inteligencia artificial basado en visión por computadora para la clasificación automática de enfermedades en plantas. Utilizando el dataset PlantVillage con 38 clases de cultivos (manzanas, uvas, tomates, maíz, entre otros), se implementó una arquitectura de transfer learning empleando MobileNetV2 pre-entrenada con ImageNet.

El modelo alcanzó una **precisión de validación del 95.02%**, un **AUC-ROC de 99.84%** y una **precisión ponderada de 95%**, demostrando una alta capacidad para distinguir entre estados saludables y patológicos en múltiples especies vegetales. El tiempo de entrenamiento fue de aproximadamente 2 horas utilizando callbacks de early stopping y reducción de learning rate.

Los resultados evidencian la efectividad del enfoque de transfer learning con redes pre-entrenadas para problemas de clasificación multiclase en el dominio agrícola, con potencial aplicación en sistemas de monitoreo de cultivos y diagnóstico fitosanitario automatizado.

---

## 1. Introducción y Descripción del Problema

### 1.1 Contexto del Proyecto

La detección temprana de enfermedades en cultivos representa uno de los desafíos más significativos en la agricultura moderna. Las pérdidas económicas asociadas a plagas y enfermedades pueden superar el 30% de la producción mundial de alimentos (Singh et al., 2022). Los métodos tradicionales de diagnóstico dependen de inspectores expertos, lo cual resulta costoso, lento y sujeto a variabilidad humana.

La visión por computadora, específicamente el deep learning, ha emergido como una herramienta poderosa para automatizar este proceso. Los modelos de redes neuronales convolucionales (CNN) pueden aprender características visuales complejas directamente de imágenes, permitiendo identificar patrones asociados a enfermedades vegetales con precisión comparable o superior a los expertos humanos (Ramcharan et al., 2019).

### 1.2 Problema Abordado

El presente proyecto aborda la clasificación multiclase de enfermedades en plantas, donde el objetivo es que un modelo computacional sea capaz de:

- Identificar la especie vegetal presente en una imagen de hoja
- Clasificar el estado fitosanitario (saludable vs. enfermo)
- Determinar el tipo específico de enfermedad, si existe

### 1.3 Objetivos

**Objetivo general:** Desarrollar un modelo de clasificación basado en visión por computadora que permita identificar enfermedades en plantas a partir de imágenes digitales.

**Objetivos específicos:**
- Implementar un modelo de deep learning con arquitectura MobileNetV2 mediante transfer learning
- Entrenar el modelo con el dataset PlantVillage conteniendo 38 clases
- Evaluar el rendimiento mediante métricas estándar de clasificación multiclase
- Generar visualizaciones que permitan interpretar el comportamiento del modelo

---

## 2. Metodología

### 2.1 Dataset y Recursos

Se utilizó el dataset **PlantVillage**, disponible públicamente a través de Kaggle (Abdallah et al., 2023). Este dataset contiene imágenes en color de hojas de plantas afectadas por diversas enfermedades, organizadas en carpetas por clase.

**Características del dataset:**

| Característica | Valor |
|----------------|-------|
| Total de imágenes | 54,305 |
| Imágenes de entrenamiento (80%) | 43,456 |
| Imágenes de validación (20%) | 10,849 |
| Número de clases | 38 |
| Tipos de cultivos | 14 |
| Tipos de enfermedades | 26 |

**Distribución de clases por cultivo:**

| Cultivo | Estados |
|---------|---------|
| Manzana | 4 (3 enfermedades + saludable) |
| Arándano | 1 (saludable) |
| Cereza | 2 (1 enfermedad + saludable) |
| Maíz | 4 (3 enfermedades + saludable) |
| Uva | 4 (3 enfermedades + saludable) |
| Naranja | 1 (enfermedad) |
| Durazno | 2 (1 enfermedad + saludable) |
| Pimiento | 2 (1 enfermedad + saludable) |
| Papa | 3 (2 enfermedades + saludable) |
| Frambuesa | 1 (saludable) |
| Soja | 1 (saludable) |
| Calabaza | 1 (enfermedad) |
| Fresa | 2 (1 enfermedad + saludable) |
| Tomate | 10 (9 enfermedades + saludable) |

### 2.2 Arquitectura del Modelo

Se empleó **MobileNetV2** como modelo base, una arquitectura diseñada por Google para ejecutarse eficientemente en dispositivos con recursos limitados. Las razones para su selección incluyen:

1. **Eficiencia computacional:** Utiliza convoluciones separables en profundidad que reducen drásticamente el número de operaciones
2. **Transfer learning:** Pre-entrenada con ImageNet (1.2M imágenes, 1000 clases)
3. **Buen balance:** Ofrece precisión aceptable con bajo consumo de recursos

**Arquitectura completa del modelo:**

```
Entrada (224 × 224 × 3)
    ↓
MobileNetV2 (convoluciones pre-entrenadas)
    ↓
GlobalAveragePooling2D (7×7×1280 → 1280)
    ↓
Dropout (0.3)
    ↓
Dense (1280 → 256) + ReLU
    ↓
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense (256 → 128) + ReLU
    ↓
Dense (128 → 38) + Softmax
    ↓
Salida (probabilidades por clase)
```

**Parámetros del modelo:**

| Capa | Parámetros |
|------|------------|
| MobileNetV2 (congelada) | 2,257,984 |
| GlobalAveragePooling2D | 0 |
| Dense_1 | 327,936 |
| BatchNormalization | 1,024 |
| Dense_2 | 32,896 |
| Dense_3 (salida) | 4,902 |
| **Total** | **2,624,742** |
| Entrenables | 366,246 |
| No entrenables | 2,258,496 |

### 2.3 Preprocesamiento de Datos

El pipeline de preprocesamiento incluyó:

1. **Redimensión:** Todas las imágenes se redimensionaron a 224×224 píxeles
2. **Normalización:** Valores de píxeles escalados de [0, 255] a [0, 1]
3. **División de datos:** 80% entrenamiento, 20% validación (aleatorio estratificado)

### 2.3.1 Justificación Técnica del Filtro Gaussiano

El filtro gaussiano es una técnica de procesamiento de imagen que aplica un kernel ponderado según la distribución normal (gaussiana) para suavizar la imagen. Se implementó con un kernel de 5×5 píxeles y desviación estándar calculada automáticamente por OpenCV.

**Fundamento matemático:**

La función gaussiana unidimensional define la ponderación de cada píxel:

$$G(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{x^2}{2\sigma^2}}$$

Para un kernel 2D, el kernel gaussiano se calcula como el producto de dos gaussianas unidimensionales:

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

**Aplicación en visión por computadora:**

```
IMAGEN ORIGINAL                DESPUÉS DEL FILTRO GAUSSIANO
┌─────────────────┐           ┌─────────────────┐
│ ▓▓░░▓▓░░▓▓░░▓▓ │           │ ░░░░░░░░░░░░░░░ │
│ ▓▓░░▓▓░░▓▓░░▓▓ │   ───►    │ ░░░░▓▓▓▓▓░░░░░ │
│ ▓▓░░▓▓░░▓▓░░▓▓ │           │ ░░░▓▓▓▓▓▓▓░░░░ │
│ ▓▓░░▓▓░░▓▓░░▓▓ │           │ ░░░░▓▓▓▓▓░░░░░ │
│ ▓▓░░▓▓░░▓▓░░▓▓ │           │ ░░░░░░░░░░░░░░░ │
└─────────────────┘           └─────────────────┘
   Ruido de alta                  Suavizado
   frecuencia
```

**Razones para su inclusión:**

| Beneficio | Descripción Técnica |
|-----------|---------------------|
| **Reducción de ruido** | El kernel gaussiano pondera los píxeles según su distancia, aplicando mayor peso a los píxeles cercanos. Esto elimina eficazmente el ruido de alta frecuencia (speckle noise, ruido de sensor) preservando las características de bordes. |
| **Preservación de bordes** | A diferencia de filtros de promediado (box blur), el kernel gaussiano otorga mayor peso al píxel central, lo que evita la pérdida de nitidez en transiciones abruptas (bordes de lesiones, manchas). |
| **Anisotropía controlada** | El kernel es simétrico en todas direcciones (isotrópico), evitando directional artifacts que podrían afectar la extracción de características por las capas convolucionales. |
| **Preparacion para segmentación** | El suavizado reduce pequeñas variaciones intra-región que pueden interferir con la detección de límites de lesiones, facilitando que la red neuronal identifique patrones de enfermedad. |

**Parámetros del kernel gaussiano (5×5):**

```
Kernel Gaussiano 5×5 (σ calculado automáticamente):
        σ = 0.3×((ksize-1)×0.5-1)+0.8 ≈ 1.1

     [1   4   7   4   1]
1/273 [4  16  26  16   4]
     [7  26  41  26   7]
     [4  16  26  16   4]
     [1   4   7   4   1]

Cada valor representa el peso del píxel vecino
en la convolución (píxel central tiene mayor peso)
```

**Elección del tamaño de kernel (5×5):**

| Tamaño | Efecto | Trade-off |
|--------|--------|-----------|
| 3×3 | Suavizado leve | Puede no eliminar suficiente ruido |
| **5×5** | **Equilibrio óptimo** | **Elimina ruido efectivamente sin perder detalles finos** |
| 7×7 | Suavizado fuerte | Riesgo de perder características pequeñas de enfermedad |
| 9×9 | Muy difuso | Detalles de enfermedad se pierden significativamente |

La elección de kernel 5×5 representa un compromiso óptimo: elimina eficientemente el ruido de sensor presente en fotografías de campo (especialmente en condiciones de poca luz) mientras preserva los detalles morfológicos característicos de las enfermedades vegetales (manchas, pudriciones, moho).

**Impacto en el modelo:**

En pruebas preliminares (sin filtro gaussiano), se observó mayor sensibilidad a variaciones de iluminación y ruido de compresión JPEG. Con el filtro gaussiano, el modelo generalizó mejor y redujo errores de clasificación en enfermedades visualmente similares.

### 2.4 Configuración del Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Optimizador | Adam |
| Learning rate inicial | 0.001 |
| Función de pérdida | Categorical Crossentropy |
| Batch size | 32 |
|Épocas máximas | 10 |
| Early stopping (patience) | 5 épocas |
| ReduceLROnPlateau (factor) | 0.5 |

### 2.5 Callbacks Implementados

Se utilizaron tres callbacks para optimizar el entrenamiento:

**EarlyStopping:** Detiene el entrenamiento cuando la pérdida de validación no mejora durante 5 épocas consecutivas, restaurando los pesos del mejor modelo.

**ReduceLROnPlateau:** Reduce el learning rate a la mitad cuando la pérdida de validación se estanca por 3 épocas, con un mínimo de 1e-7.

**ModelCheckpoint:** Guarda automáticamente el modelo cuando se logra una mejora en la precisión de validación.

---

## 3. Resultados del Modelo

### 3.1 Métricas de Rendimiento

**Resultados finales del entrenamiento (época 9 - mejor modelo):**

| Métrica | Entrenamiento | Validación |
|---------|---------------|------------|
| Accuracy | 92.84% | 95.02% |
| Loss | 0.207 | 0.148 |
| Precision | 94.34% | 95.84% |
| Recall | 91.78% | 94.39% |
| AUC-ROC | 99.80% | 99.84% |
| Top-3 Accuracy | 99.13% | 99.43% |

### 3.2 Análisis por Clase

El modelo demuestra un rendimiento generalmente alto en la mayoría de las 38 clases. A continuación se destacan las clases con mejor y peor desempeño:

**Clases con mejor F1-score:**
- Corn_healthy: 1.00
- Orange_Haunglongbing: 1.00
- Grape_Leaf_blight: 0.99
- Blueberry_healthy: 0.99
- Squash_Powdery_mildew: 0.99
- Strawberry_Leaf_scorch: 0.99
- Soybean_healthy: 0.99

**Clases con menor F1-score:**
- Potato_healthy: 0.74 (30 samples - desbalance)
- Tomato_Early_blight: 0.72
- Tomato_Target_Spot: 0.79
- Corn_Cercospora: 0.80

### 3.3 Curvas de Aprendizaje

El análisis de las curvas de aprendizaje revela:

1. **Convergencia estable:** El modelo converge rápidamente, alcanzando >90% de accuracy en la primera época
2. **Generalización adecuada:** Las métricas de validación superan ligeramente a las de entrenamiento, indicando buen generalization
3. **Mejora continua:** La pérdida de validación disminuye consistentemente hasta la época 9
4. **Ausencia de overfitting:** No se observa divergencia significativa entre curvas de entrenamiento y validación

### 3.4 Matriz de Confusión

La matriz de confusión normalizada muestra:

- **Alta precisión en la diagonal:** El modelo clasifica correctamente la mayoría de las instancias
- **Confusiones principales:**
  - Tomato_Early_blight ↔ Tomato_Late_blight (enfermedades visuales similares)
  - Potato_healthy ↔ Potato_Early_blight (ambos tonos marrones)
  - Tomato_Target_Spot ↔ otras enfermedades del tomate

### 3.5 Tiempo de Entrenamiento

| Métrica | Valor |
|---------|-------|
| Tiempo total | 7,450.14 segundos (~2.07 horas) |
| Tiempo por época | ~745 segundos (~12.4 minutos) |
| Pasos por época | 1,358 batches |

---

## 4. Análisis Interpretativo

### 4.1 Rendimiento General

El modelo logra una **precisión de clasificación del 95.02%** en el conjunto de validación, lo cual representa un desempeño excelente para un problema de clasificación multiclase con 38 categorías. Este resultado es consistente con estudios previos que reportan precisiones entre 90-99% en el dataset PlantVillage (Mohanty et al., 2016; Too et al., 2019).

### 4.2 Análisis de Errores

Las principales fuentes de error identificadas incluyen:

1. **Similitud visual entre enfermedades:** Algunas enfermedades producen patrones visuales similares (e.g., diferentes tipos de manchas foliares)
2. **Desbalance de clases:** Clases con pocas muestras (Potato_healthy: 30) muestran menor rendimiento
3. **Variabilidad en iluminación:** El dataset incluye imágenes con diferentes condiciones de luz y fondo

### 4.3 Fortalezas del Modelo

- **Transfer learning efectivo:** El uso de MobileNetV2 pre-entrenada permite aprovechar características visuales generalizadas
- **Regularización adecuada:** Los dropout layers (30%) y early stopping previenen el sobreajuste
- **Batch normalization:** Estabiliza el entrenamiento y mejora la convergencia
- **AUC-ROC excepcional (99.84%):** Indica excelente capacidad discriminativa entre clases

### 4.4 Limitaciones

1. **Dependencia de fondo controlado:** El modelo fue entrenado con imágenes de fondo uniforme (hojas recortadas)
2. **Contexto limitado:** No se proporciona información sobre la planta completa, solo la hoja
3. **Clases desbalanceadas:** Algunas enfermedades están subrepresentadas
4. **Condiciones específicas:** Imágenes adquiridas bajo condiciones controladas de iluminación

### 4.5 Consideraciones Éticas

La implementación de sistemas de diagnóstico automatizado en agricultura debe considerar:

- La importancia de la supervisión humana en decisiones críticas
- La necesidad de validación continua con datos del campo
- Los sesgos potenciales derivados de datasets de entrenamiento
- El impacto en comunidades agrícolas y trabajadores del sector

---

## 5. Aplicaciones y Relevancia

### 5.1 Aplicaciones Potenciales

El modelo desarrollado tiene aplicabilidad en:

1. **Sistemas de monitoreo de cultivos:** Integración en drones o cámaras de campo para detección temprana
2. **Aplicaciones móviles:** Herramientas de diagnóstico accesibles para agricultores
3. **Sistemas de alerta temprana:** Notificaciones automáticas ante detección de enfermedades
4. **Investigación agrícola:** Apoyo en estudios de epidemiología vegetal

### 5.2 Comparación con Estado del Arte

| Estudio | Modelo | Precisión |
|---------|--------|-----------|
| Mohanty et al. (2016) | AlexNet | 97.7% |
| Too et al. (2019) | DenseNet161 | 99.7% |
| Presente trabajo | MobileNetV2 | 95.0% |

Aunque arquitecturas más profundas como DenseNet161 logran mayor precisión, MobileNetV2 ofrece ventajas en eficiencia computacional, haciéndola más adecuada para implementación en dispositivos edge.

---

## 6. Conclusiones

### 6.1 Síntesis de Resultados

El presente proyecto demuestra la viabilidad de utilizar transfer learning con MobileNetV2 para la clasificación automática de enfermedades en plantas. Los resultados obtenidos, con una precisión de validación del 95.02% y AUC-ROC de 99.84%, confirman la efectividad del enfoque para resolver problemas de clasificación multiclase en el dominio agrícola.

### 6.2 Retos Técnicos Superados

- Implementación exitosa de pipeline de datos con generadores de TensorFlow
- Optimización del entrenamiento mediante callbacks adecuados
- Manejo eficiente de dataset de más de 54,000 imágenes
- Prevención efectiva del sobreajuste mediante técnicas de regularización

### 6.3 Recomendaciones Futuras

1. **Expansión del dataset:** Incluir imágenes con fondos variados, diferentes condiciones de iluminación y estadios tempranos de enfermedad
2. **Data augmentation avanzado:** Rotaciones, cambios de perspectiva y simulaciones de condiciones ambientales
3. **Ensemble de modelos:** Combinar múltiples arquitecturas para mejorar la robustez
4. **Segmentación:** Implementar detección de regiones afectadas dentro de las hojas
5. **Despliegue en campo:** Validar el modelo con imágenes capturadas en condiciones reales

### 6.4 Reflexión Final

Este proyecto integra conceptos fundamentales de visión por computadora, deep learning y desarrollo de modelos de machine learning, aplicando Transfer Learning para resolver un problema real con impacto agrícola. El proceso de diseñar, entrenar y evaluar el modelo fortalece competencias técnicas en procesamiento de imágenes, diseño de arquitecturas neuronales y análisis de métricas de rendimiento.

La comunicación efectiva de resultados técnicos mediante este informe refleja la importancia de la divulgación científica y la transferencia de conocimiento en el campo de la inteligencia artificial aplicada.

---

## 7. Referencias Bibliográficas

Abdallah, A., Ali, A., & Ramdan, M. (2023). *PlantVillage dataset* (Version 3) [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Contreras-Bravo, L., Fuentes-López, H., & Rodríguez-Molano, J. (2024). *Algoritmos supervisados y de ensamble con Python: Implementación y estrategia de optimización*. Ediciones de la U.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science, 7*, 1419.

Ramcharan, A., Baranowski, K., McCloskey, P., Ahmed, B., Legg, J., & Hughes, D. P. (2019). Deep learning for image-based cassava disease detection. *Frontiers in Plant Science, 10*, 1182.

Rosebrock, A. (2019). *Deep Learning for Computer Vision with Python*. Apress.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 4510-4520.

Sharda, R., Dursun, R., & Turban, E. (2021). *Analytics, Data Science, Artificial Intelligence: Systems for Decision Support* (11th ed.). Pearson Educación.

Singh, A., Ganapathysubramanian, B., Singh, A. K., & Kumar, S. (2022). Machine learning for high-throughput stress phenotyping in plants. *Trends in Plant Science, 21*(2), 110-124.

Too, E. C., Yujian, L., Njuki, S., & Ying-chun, L. (2019). A comparative study of fine-tuning deep learning models for plant disease identification. *Computers and Electronics in Agriculture, 161*, 272-279.

---

## Anexos

### Anexo A: Estructura del Código

El código desarrollado se organiza en las siguientes secciones:

1. Importación de librerías (TensorFlow, OpenCV, NumPy, Matplotlib)
2. Configuración de hiperparámetros
3. Definición de funciones de preprocesamiento
4. Creación de generadores de datos
5. Carga y configuración del modelo base
6. Definición de la arquitectura personalizada
7. Compilación del modelo con métricas
8. Configuración de callbacks
9. Entrenamiento del modelo
10. Evaluación y análisis de métricas
11. Generación de visualizaciones
12. Exportación del modelo

### Anexo B: Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `best_model.keras` | Modelo entrenado guardado |
| `confusion_matrix.png` | Visualización de la matriz de confusión |
| `accuracy_loss_curves.png` | Curvas de accuracy y loss |
| `precision_recall_curves.png` | Curvas de precision y recall |
| `auc_curves.png` | Curva AUC-ROC |
| `metricas_entrenamiento.json` | Resumen de métricas en formato JSON |
| `clases.json` | Lista de nombres de clases |

---

*Documento generado como parte de la actividad formativa del curso de Visión por Computador*
