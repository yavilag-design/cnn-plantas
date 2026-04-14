# GUION PARA VIDEO DE 3 MINUTOS

## PROYECTO: CLASIFICACIÓN DE ENFERMEDADES EN PLANTAS MEDIANTE VISIÓN POR COMPUTADOR Y DEEP LEARNING

---

### ESCENA 1: INTRODUCCIÓN Y CONTEXTO

[Duración: 30 segundos]

La agricultura moderna enfrenta uno de sus desafíos más significativos: la detección temprana de enfermedades en cultivos. Las pérdidas económicas asociadas a plagas y enfermedades pueden superar el 30% de la producción mundial de alimentos, afectando la seguridad alimentaria global. Los métodos tradicionales de diagnóstico dependen de inspectores expertos, siendo procesos lentos, costosos y sujetos a variabilidad humana.

En este proyecto, desarrollamos un sistema inteligente basado en visión por computadora y deep learning que permite clasificar automáticamente enfermedades en plantas a partir de imágenes digitales de hojas. Utilizando técnicas de transfer learning con la arquitectura MobileNetV2, logramos un modelo con una precisión de validación del 95.02% y un AUC-ROC del 99.84%, demostrando la viabilidad de esta tecnología para resolver problemas reales en el dominio agrícola.

---

### ESCENA 2: DATASET Y RECURSOS

[Duración: 20 segundos]

Para el entrenamiento del modelo utilizamos el dataset PlantVillage, disponible públicamente a través de Kaggle. Este dataset contiene 54,305 imágenes de hojas de plantas afectadas por diversas enfermedades, organizadas en 38 clases diferentes. La distribución de datos se dividió en 43,456 imágenes para entrenamiento, representando el 80% del conjunto, y 10,849 imágenes para validación.

El dataset abarca 14 tipos de cultivos diferentes, incluyendo manzana, uva, tomate, maíz, papa, fresa, cereza, entre otros. Cada cultivo puede presentar múltiples estados: saludables o afectados por diferentes enfermedades, alcanzando hasta 10 estados diferentes para el caso del tomate.

---

### ESCENA 3: ARQUITECTURA DEL MODELO

[Duración: 30 segundos]

Implementamos una arquitectura basada en transfer learning utilizando MobileNetV2 como modelo base, pre-entrenada con el dataset ImageNet que contiene 1.2 millones de imágenes y 1000 clases. MobileNetV2 fue diseñada por Google para ejecutarse eficientemente en dispositivos con recursos limitados, utilizando convoluciones separables en profundidad que reducen drásticamente el número de operaciones computacionales.

La arquitectura completa del modelo consiste en: la red MobileNetV2 congelada que actúa como extractor de características, seguida de una capa de GlobalAveragePooling que reduce los mapas de características de 7x7x1280 a un vector de 1280 características. Posteriormente, se aplicó una capa de Dropout con rate del 30% para regularización, una capa densa de 256 neuronas con activación ReLU, BatchNormalization, otra capa de Dropout del 30%, una segunda capa densa de 128 neuronas, y finalmente la capa de salida con 38 neuronas y activación softmax para la clasificación multiclase.

El modelo cuenta con aproximadamente 2.6 millones de parámetros en total, de los cuales solo 366,000 son entrenables, mientras que los 2.2 millones de MobileNetV2 permanecen congelados aprovechando el conocimiento previo aprendido.

---

### ESCENA 4: PREPROCESAMIENTO - FILTRO GAUSSIANO

[Duración: 20 segundos]

Como parte fundamental del preprocesamiento de imágenes, implementamos un filtro gaussiano con un kernel de 5x5 píxeles. Este filtro aplica una distribución gaussiana para ponderar los píxeles vecinos durante la convolución, otorgando mayor peso a los píxeles más cercanos al centro.

El filtro gaussiano cumple múltiples propósitos técnicos: reduce efectivamente el ruido de alta frecuencia presente en fotografías de campo, especialmente en condiciones de poca luz; preserva los bordes de las lesiones y manchas gracias a su ponderación central; y es isotrópico, evitando artifacts direccionales que podrían afectar la extracción de características.

La elección del kernel de 5x5 representa un compromiso óptimo entre eliminación de ruido y preservación de detalles morfológicos característicos de las enfermedades vegetales, como manchas, pudriciones y moho.

---

### ESCENA 5: RESULTADOS

[Duración: 30 segundos]

Los resultados del entrenamiento fueron altamente satisfactorios. El modelo alcanzó una precisión de validación del 95.02%, superando la precisión de entrenamiento del 92.84%, lo cual indica un excelente generalization sin presencia de overfitting. La precisión ponderada alcanzó el 95% y el recall ponderado el 94.39%.

La métrica más destacada fue el AUC-ROC, que alcanzó un 99.84% en validación, indicando una capacidad discriminativa excepcional del modelo para distinguir entre las 38 clases de enfermedades y estados saludables. La precisión Top-3, que mide si la clase correcta se encuentra entre las tres predicciones más probables, llegó al 99.43%.

En cuanto al rendimiento por clase, el modelo demostró un desempeño excelente en la mayoría de las categorías. Clases como Maíz saludable, Naranja con Huanglongbing, Uva con Leaf blight, y Arándano saludable alcanzaron F1-scores de 1.00 o 0.99. Las clases con menor rendimiento fueron Papa saludable con 0.74 y Tomate con Early blight con 0.72, principalmente debido al desbalance de datos en el dataset.

El tiempo total de entrenamiento fue de aproximadamente 2 horas y 7 minutos, utilizando early stopping que detuvo el proceso en la época 9 de 10 máximas permitidas.

---

### ESCENA 6: APLICACIONES

[Duración: 15 segundos]

Este modelo tiene múltiples aplicaciones potenciales en el sector agrícola. Puede integrarse en sistemas de monitoreo de cultivos mediante drones para detección temprana de focos de enfermedad, desplegarse en aplicaciones móviles accesibles para agricultores en campo, implementarse en sistemas industriales de control de calidad para clasificación automatizada de productos, y utilizarse en investigación agrícola para estudios epidemiológicos y monitoreo de resistencia a fungicidas.

---

### ESCENA 7: CONCLUSIONES

[Duración: 20 segundos]

Para finalizar, este proyecto demuestra la viabilidad de utilizar transfer learning con MobileNetV2 para la clasificación automática de enfermedades en plantas. Los resultados obtenidos, con una precisión de validación del 95.02% y AUC-ROC de 99.84%, confirman la efectividad del enfoque para resolver problemas de clasificación multiclase en el dominio agrícola.

El filtro gaussiano en el preprocesamiento demostró ser una técnica valiosa que mejora la robustez del modelo frente a ruido de sensor y variaciones de iluminación. El uso de transfer learning no solo acelera el desarrollo sino que también mejora los resultados al aprovechar características visuales generalizadas pre-entrenadas.

Como trabajo futuro, se propone expandir el dataset con imágenes de fondos naturales, implementar data augmentation avanzado, explorar arquitecturas de ensemble, y validar el modelo en condiciones reales de campo para facilitar su despliegue en dispositivos edge.

---

### NOTAS DE PRODUCCIÓN

- Velocidad de lectura recomendada: 150-160 palabras por minuto
- Tiempo total estimado: 3 minutos exactos
- Tono: profesional pero accesible
- Evitar muletillas como "bueno", "entonces", "así que"
- Pausas breves entre secciones para transición
