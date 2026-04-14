# Documentación Técnica: Arquitectura del Modelo y Métricas

---

## 1. ARQUITECTURA DEL MODELO

### 1.1 Modelo Base: MobileNetV2

**MobileNetV2** es una arquitectura de red neuronal convolucional diseñada por Google para ejecutarse eficientemente en dispositivos móviles. Fue introducida en 2018 y se basa en dos conceptos clave:

#### 1.1.1 Depthwise Separable Convolutions

A diferencia de las convoluciones estándar que aplican filtros a todas los canales de entrada simultáneamente, MobileNetV2 utiliza **convoluciones separables en profundidad**:

```
CONVOLUCIÓN ESTÁNDAR (1x1):
- Para una imagen de 224x224x3 (RGB)
- Un filtro 3x3 procesa los 3 canales -> produce 1 canal de salida
- Con 1280 filtros de salida -> 1280 canales

Costo computacional = H × W × C × K × K × M
(H=alto, W=ancho, C=canales entrada, K=tamaño kernel, M=canales salida)

CONVOLUCIÓN DEPTHWISE SEPARABLE (2 pasos):
1. Depthwise: cada canal se procesa independientemente
   - 3 filtros 3x3, uno por canal -> 3 canales
2. Pointwise: combiner usando convoluciones 1x1
   - Convierte los 3 canales a los 1280 de salida

Costo = (H × W × C × K × K) + (H × W × C × M)
```

**Ahorro**: Aproximadamente 8-9x menos operaciones que una convolución estándar.

#### 1.1.2 Bloques Inverted Residuals with Linear Bottlenecks

MobileNetV2 usa bloques especiales llamados **Inverted Residuals**:

```
ARQUITECTURA DEL BLOQUE:
1. Expansión (1x1 conv): canales 3 → 64 (expande 6x)
2. Depthwise (3x3 conv): procesa cada canal independientemente
3. Proyección (1x1 conv): canales 64 → 3 (reduce)
4. Skip connection: si dimensiones son iguales

         ┌──────────────┐
    ────>│ 1x1 Conv     │──> Expande canales
         │ (ReLU6)      │
         ├──────────────┤
         │ Depthwise    │──> Convolución espacial
         │ 3x3         │
         ├──────────────┤
         │ 1x1 Conv    │──> Reduce canales
         │ (Linear)    │──> Sin activación final
         ┴──────────────┘
              │
              └──> Skip connection (si dims iguales)
```

**Por qué lineal?**: Aplicar ReLU después de comprimir destruye información. El bottleneck final usa activación lineal para preservar la información.

#### 1.1.3 Especificaciones de MobileNetV2

| Capa/Parámetro | Valor |
|----------------|-------|
| Input | 224×224×3 |
| Salida | 7×7×1280 |
| Parámetros | 2,257,984 (~3.4 MB) |
| Profundidad | 53 capas |

**Por qué MobileNetV2?**
- Pre-entrenado con ImageNet (1.2M imágenes, 1000 clases)
- Transfer learning: aprovecha Features aprendidos
- Eficiente y preciso

---

### 1.2 Capas Añadidas sobre MobileNetV2

```python
model = Sequential([
    # Modelo base pre-entrenado
    base_model,                           # MobileNetV2
    
    # Global Average Pooling
    GlobalAveragePooling2D(),            # 7x7x1280 → 1x1x1280
    
    # Dropout 1
    Dropout(0.3),                        # 30% de neuronas se apagan
    
    # Dense 1
    Dense(256, activation='relu'),        # 1280 → 256
    
    # Batch Normalization
    BatchNormalization(),                # Normaliza activaciones
    
    # Dropout 2
    Dropout(0.3),                        # 30% de neuronas se apagan
    
    # Dense 2
    Dense(128, activation='relu'),        # 256 → 128
    
    # Capa de salida
    Dense(38, activation='softmax')       # 128 → 38 clases
])
```

#### 1.2.1 GlobalAveragePooling2D

Reduce cada mapa de características 7x7 a un solo valor:

```
Entrada:  tensor de forma (batch, 7, 7, 1280)
Salida:   tensor de forma (batch, 1280)

EJEMPLO:
[[[1, 2, 3],    →   promedio de todo el mapa
  [4, 5, 6]],
 [[7, 8, 9],
  [10,11,12]]]

= 6.5 (promedio de todos los valores)
```

**Ventajas**:
- Reduce drásticamente parámetros (de ~2.5M a ~2.5k)
- Más robusto a translaciones
- Evita overfitting

#### 1.2.2 Dropout

Durante el entrenamiento, aleatoriamente "apaga" neuronas:

```
DROPOUT(0.3):
- 30% de las neuronas se temporariamente "apagan"
- Las salida de esas neuronas = 0
- Durante inferencia, todas las neuronas activas pero escaladas por (1-rate)

         ENTRENAMIENTO          INFERENCIA
Entrada: [1, 2, 3, 4, 5]   →   [1, 2, 3, 4, 5] × 0.7
         ↓ (30% drop)              ↓
         [1, 0, 3, 0, 5]
```

**Por qué funciona?**
- Previene que las neuronas dependan demasiado de otras
- Fuerza redundancia
- Simula un "ensemble" de redes

#### 1.2.3 Dense (Fully Connected)

Cada neurona se conecta a todas las anteriores:

```
Entrada: 1280 neuronas
Salida:  256 neuronas
Parámetros: 1280 × 256 + 256 = 327,936

           Pesos (W)          Bias (b)
           ┌─────────┐       ┌───┐
   x₁ ────┤         │
   x₂ ────┤    W    │──────>├───┤───> y₁ = ReLU(Σwᵢxᵢ + b)
   x₃ ────┤  1280x256│      │   │
   ... ───┤         │      │   │
   x₁₂₈₀────┤         │      │   │
           └─────────┘       └───┘
```

**Función de activación ReLU**:
```
ReLU(x) = max(0, x)

ReLU(-5) = 0
ReLU(0)  = 0
ReLU(5)  = 5
```

**Ventajas de ReLU**:
-简单 (simple de calcular)
- No sufren de "vanishing gradient"
- Introduce sparsity (algunas neuronas están inactivas)

#### 1.2.4 BatchNormalization

Normaliza las activaciones de cada batch:

```
INPUT: batch de activations con media μ y desviación estándar σ
OUTPUT: activations normalizadas

Pasos:
1. Normalizar: (x - μ) / σ
2. Escalar: γ × normalized
3. Desplazar: + β

Parámetros aprendibles: γ (gamma) y β (beta)

EJEMPLO:
x = [1, 2, 3, 4, 5]
μ = 3
σ = 1.41
Normalizado: [-1.41, -0.70, 0, 0.70, 1.41]
Escalado: γ=1, β=0 → mismo resultado
```

**¿Por qué funciona?**
- Estabiliza el entrenamiento
- Permite learning rates más altos
- Regulariza ligeramente
- Reduce internal covariate shift

#### 1.2.5 Capa de Salida (Softmax)

Convierte logits a probabilidades (suman 1):

```
SOFTMAX para 38 clases:
z = [z₁, z₂, ..., z₃₈]
σ(zᵢ) = eᶻᵢ / Σⱼ eᶻʲ

EJEMPLO (3 clases):
z = [2.0, 1.0, 0.1]
eᶻ = [7.39, 2.72, 1.11]
suma = 11.22
softmax = [0.659, 0.242, 0.099]

Interpretación:
- Clase 1: 65.9% de probabilidad
- Clase 2: 24.2%
- Clase 3: 9.9%
```

**¿Por qué softmax?**
- Salida es interpretable como probabilidad
- Útil para clasificación multiclase
- Diferenciable

---

## 2. OPTIMIZACIÓN

### 2.1 Optimizer: Adam

**Adam** = Adaptive Moment Estimation

Combina:
- Momentum: acelera en direcciones consistentas
- RMSProp: adapta learning rate por parámetro

```
ALGORITMO:
1. Inicializar:
   m = 0 (primer momento)
   v = 0 (segundo momento)
   t = 0 (contador)

2. Por cada batch:
   t ← t + 1
   
   # Gradientes
   g ← ∇θ L(θ)
   
   # Actualizar momentos (exponential moving average)
   m ← β₁ × m + (1 - β₁) × g        # Momentum
   v ← β₂ × v + (1 - β₂) × g²       # RMSProp
   
   # Bias correction
   m̂ ← m / (1 - β₁ᵗ)
   v̂ ← v / (1 - β₂ᵗ)
   
   # Actualizar pesos
   θ ← θ - α × m̂ / (√v̂ + ε)

HIPERPARÁMETROS:
- α (learning rate) = 0.001
- β₁ = 0.9 (momentum)
- β₂ = 0.999 (RMSProp)
- ε = 10⁻⁷ (numerical stability)
```

**¿Por qué Adam?**
- Robusto a diferentes escalas de features
- Requires less tuning
- Funciona bien por defecto

### 2.2 Función de Pérdida: Categorical Crossentropy

Para clasificación multiclase, calculamos la distancia entre distribuciones:

```
CROSS-ENTROPY:
L = -Σᵢ yᵢ × log(ŷᵢ)

Donde:
yᵢ = probabilidad real (one-hot: 1 en clase correcta, 0 en otras)
ŷᵢ = probabilidad predicha por softmax

EJEMPLO (clase correcta = 1):
y = [1, 0, 0]
ŷ = [0.7, 0.2, 0.1]
L = -[1×log(0.7) + 0×log(0.2) + 0×log(0.1)]
   = -log(0.7)
   = 0.3567

COMPARANDO PREDICCIONES:
ŷ₁ = [0.99, 0.01, 0.00] → L = 0.0101 (MUY BUENO)
ŷ₂ = [0.33, 0.33, 0.34] → L = 1.0986 (MUY MALO)
```

**¿Por qué cross-entropy?**
- Penaliza fortemente predicciones incorrectas confidentes
- Tiende a producir probabilidades extremas
- Works well with softmax

---

## 3. CALLBACKS

### 3.1 EarlyStopping

Detiene el entrenamiento cuando la pérdida de validación deja de mejorar.

```
PARAMETROS:
- monitor = 'val_loss'      # Qué monitorear
- patience = 5              # Épocas sin mejora antes de parar
- restore_best_weights = True  # Restaurar mejores pesos
- verbose = 1              # Mensajes

EJEMPLO (patience=5):
Epoch 1: val_loss = 0.250 (best: 0.250)
Epoch 2: val_loss = 0.200 (best: 0.200) ✓ nueva mejor
Epoch 3: val_loss = 0.190 (best: 0.190) ✓
Epoch 4: val_loss = 0.195 (sin mejora 1/5)
Epoch 5: val_loss = 0.220 (sin mejora 2/5)
Epoch 6: val_loss = 0.210 (sin mejora 3/5)
Epoch 7: val_loss = 0.205 (sin mejora 4/5)
Epoch 8: val_loss = 0.215 (sin mejora 5/5) → PARAR
        Se restauran pesos de Epoch 3 (val_loss = 0.190)
```

**Propósito:**
- Previene overfitting
- Ahorra tiempo computacional
- Evita degradación del modelo

### 3.2 ReduceLROnPlateau

Reduce el learning rate cuando la pérdida se estanca.

```
PARAMETROS:
- monitor = 'val_loss'
- factor = 0.5              # Multiplicar LR por 0.5
- patience = 3               # Épocas sin mejora
- min_lr = 1e-7             # Learning rate mínimo
- verbose = 1

EJEMPLO:
LR inicial = 0.001
Epoch 1: val_loss = 0.250  → LR = 0.001
Epoch 2: val_loss = 0.200  ✓ LR = 0.001
Epoch 3: val_loss = 0.190  ✓ LR = 0.001
Epoch 4: val_loss = 0.200  (sin mejora 1/3)
Epoch 5: val_loss = 0.210  (sin mejora 2/3)
Epoch 6: val_loss = 0.205  (sin mejora 3/3) → LR = 0.0005
Epoch 7: val_loss = 0.195  ✓ LR = 0.0005
...
Epoch 9: val_loss = 0.190  (sin mejora 3/3) → LR = 0.00025
```

**Propósito:**
- Fine-tuning cuando el modelo se estanca
- Convergencia más estable
- Escapa de óptimos locales

### 3.3 ModelCheckpoint

Guarda el modelo cuando mejoran las métricas.

```
PARAMETROS:
- 'best_model.keras'         # Path del archivo
- monitor = 'val_accuracy'
- save_best_only = True     # Solo guardar si mejora
- verbose = 1

EJEMPLO:
Epoch 1: val_acc = 0.918 → guardar (best so far)
Epoch 2: val_acc = 0.931 → guardar (nueva mejor)
Epoch 3: val_acc = 0.940 → guardar (nueva mejor)
Epoch 4: val_acc = 0.939 → NO guardar (no mejoró)
Epoch 5: val_acc = 0.945 → guardar (nueva mejor)
...
```

**Propósito:**
- Preserva el mejor modelo
- Permite recuperación post-entrenamiento
- No guardar modelos intermedios (ahorra espacio)

---

## 4. MÉTRICAS

### 4.1 Accuracy

Porcentaje de predicciones correctas.

```
ACCURACY = predicciones_correctas / total_predicciones

EJEMPLO:
Predicciones: [1, 2, 3, 4, 5]
Reales:      [1, 2, 2, 4, 5]
Correctas:   [✓, ✓, ✗, ✓, ✓] = 4/5 = 0.80 = 80%

LIMITACIONES:
- No distingue entre tipos de error
- Engañoso con datasets desbalanceados
- 99% accuracy puede ser malo si el 1% es crítico
```

### 4.2 Precision

De las predicciones positivas, ¿cuántas son realmente positivas?

```
PRECISION = TP / (TP + FP)

EJEMPLO (detección de enfermedad):
- Predicciones positivas: 100
- Verdaderos positivos (TP): 80
- Falsos positivos (FP): 20

Precision = 80 / (80 + 20) = 80%

INTERPRETACIÓN:
"Si el modelo dice que hay enfermedad, está correcto 80% de las veces"
```

### 4.3 Recall (Sensitivity)

De todas lasInstances reales positivas, ¿cuántas detectamos?

```
RECALL = TP / (TP + FN)

EJEMPLO:
- Casos reales con enfermedad: 120
- Verdaderos positivos (TP): 80
- Falsos negativos (FN): 40

Recall = 80 / (80 + 40) = 66.67%

INTERPRETACIÓN:
"El modelo detecta 66.67% de los casos de enfermedad"
```

### 4.4 F1-Score

Media armónica de Precision y Recall.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

EJEMPLO:
Precision = 0.80
Recall    = 0.6667

F1 = 2 × (0.80 × 0.6667) / (0.80 + 0.6667)
   = 1.0667 / 1.4667
   = 0.727

¿POR QUÉ MEDIA ARMÓNICA?
- Penaliza cuando hay desbalance entre P y R
- Si P=1.0 y R=0.1 → media aritmética = 0.55 (engañoso)
-                             armónica = 0.18 (honesto)
```

### 4.5 AUC-ROC

**Area Under the Curve - Receiver Operating Characteristic**

Mide la capacidad del modelo de distinguir entre clases.

```
CURVA ROC:
- Eje Y: True Positive Rate (TPR) = TP/(TP+FN) = Recall
- Eje X: False Positive Rate (FPR) = FP/(FP+TN)

EJEMPLO:
Threshold=0.9: TPR=0.4, FPR=0.05
Threshold=0.7: TPR=0.7, FPR=0.10
Threshold=0.5: TPR=0.85, FPR=0.20
Threshold=0.3: TPR=0.95, FPR=0.35
Threshold=0.1: TPR=1.0, FPR=0.60

AUC = área bajo la curva ROC

INTERPRETACIÓN:
- AUC = 1.0: Perfecto (separa perfectamente las clases)
- AUC = 0.5: Aleatorio (como lanzar moneda)
- AUC = 0.9: Muy bueno

EJEMPLO VISUAL:
         TPR
          │
    1.0   │    ╱-perfecto (AUC=1.0)
          │   ╱ 
    0.8   │  ╱   ╲-bueno (AUC=0.9)
          │ ╱     ╲
    0.6   │╱       ╲
          │          ╲-regular (AUC=0.7)
    0.4   ├───────────╲
          │             ╲-pobre (AUC=0.6)
    0.2   ├─────────────╲
          │               ��
    0.0   └───────────────┼──── FPR
         0        0.5    1.0
```

### 4.6 Top-K Accuracy

La predicción correcta está entre las K predicciones más probables.

```
TOP-3 ACCURACY:
- Si la clase real es la #1 predicha → ✓
- Si la clase real es la #2 predicha → ✓
- Si la clase real es la #3 predicha → ✓

EJEMPLO:
Clase real: Manzana
Top 3 predicciones: [Manzana, Pera, Uva] → ✓ (1/3 = correcto)
Top 3 predicciones: [Pera, Manzana, Uva] → ✓ 
Top 3 predicciones: [Pera, Uva, Manzana] → ✓

PROPÓSITO:
- Más tolerate a errores menores
- Útil cuando hay muchas clases
- Mide "qué tan cerca" estuvo la respuesta
```

---

## 5. CLASSIFICATION REPORT

Reporte detallado por cada clase:

```
EJEMPLO DE SALIDA:
                  precision  recall  f1-score   support

       Apple___      0.95      0.92     0.93      1500
          Apple___rot   0.88      0.90     0.89      1200
            Cherry_     0.97      0.96     0.96      1800
         ...
           accuracy                          0.93     54305
          macro avg       0.92     0.91     0.91     54305
       weighted avg       0.93     0.91     0.92     54305

CADA COLUMNA:
- precision: ¿Qué tan precisas son mis predicciones para esta clase?
- recall: ¿Cuántas de las Instances reales de esta clase detecté?
- f1-score: Media armónica de ambas
- support: Número de samples de esa clase

AGREGADOS:
- accuracy: Overall accuracy (predicciones correctas / total)
- macro avg: Promedio simple de todas las clases
- weighted avg: Promedio pesado por support de cada clase
```

### 5.1 Cómo Interpretar

```
CLASE "Apple___healthy" (saludable):
precision=0.95 → Cuando predije "saludable", estuve correcto 95%
recall=0.92 → Detecté 92% de todas las hojas saludables
f1=0.93 → Balance entre precision y recall

¿CUÁL IMPLEMENTAR?
- Precision importante: Minimizar falsos positivos
  → No quiero decir "enferma" cuando está sana
- Recall importante: Minimizar falsos negativos  
  → No quiero decir "sana" cuando está enferma
- F1: Balance cuando ambas son importantes
```

---

## 6. MATRIZ DE CONFUSIÓN

Tabla que muestra las combinaciones de valores reales vs predichos.

```
EJEMPLO (3 clases simplificado):
                    PREDICHO
                  Saludable  Enferma  Virus
     ┌──────────┬─────────┬────────┐
S     │  1450   │   45    │   5    │  ← 1500 reales saludables
     ├──────────┼─────────┼────────┤
R  E  │   30    │  1170   │  100   │  ← 1300 reales enfermas
     ├──────────┼─────────┼────────┤
A  V  │   10    │   85    │  1105  │  ← 1200 reales virus
     └──────────┴─────────┴────────┘
     
INTERPRETACIÓN:
- Diagonal: Predicciones correctas (1450 + 1170 + 1105 = 3725)
- Fuera de diagonal: Errores

ERRORES COMUNES:
- Saludable → Enferma (30): Falso positivo de enfermedad
- Enferma → Virus (100): Confusión entre enfermedades
- Virus → Enferma (85): Confusión entre enfermedades
```

### 6.1 Matriz Normalizada

Para comparar entre clases con diferente número de samples:

```
NORMALIZADA (por fila):
                  PREDICHO
               Saludable  Enferma  Virus
     ┌────────┬─────────┬────────┐
S     │ 0.967 │  0.030 │  0.003 │
     ├────────┼─────────┼────────┤
E     │ 0.023 │  0.900 │  0.077 │
     ├────────┼─────────┼────────┤
V     │ 0.008 │  0.071 ���  0.921 │
     └────────┴─────────┴────────┘

Cada fila suma a 1.0 (100%)
```

### 6.2 Patrones de Error

```
PATRÓN IDEAL: diagonal = 1.0, resto = 0

PATRONES PROBLEMÁTICOS:

1. много falsos en una clase:
   → Esa clase se confunde con otras
   
2. Confusión sistemática entre 2 clases:
   → Posiblemente son muy similares visualmente
   → Necesita más datos de esas clases
   → Necesita data augmentation específico

3. Baja precision general:
   → El modelo no está aprendiendo bien
   → Puede necesitar más épocas
   → Puede necesitar arquitectura diferente
```

---

## 7. RESUMEN DE FLUJO DE DATOS

```
INPUT IMAGE (224×224×3 RGB)
         ↓
    MobileNetV2
    (convoluciones + pooling)
         ↓
    Feature Maps (7×7×1280)
         ↓
    GlobalAveragePooling2D
    (promedio espacial)
         ↓
    Vector (1280)
         ↓
    Dropout(0.3)
         ↓
    Dense(256) + ReLU
         ↓
    BatchNorm
         ↓
    Dropout(0.3)
         ↓
    Dense(128) + ReLU
         ↓
    Dense(38) + Softmax
         ↓
    PROBABILIDADES POR CLASE (38)
```

---

## 8. TABLA RESUMEN DE MÉTRICAS

| Métrica | Rango Óptimo | Significado |
|---------|-------------|-----------|
| Accuracy | 0-1 (mayor mejor) | Porcentaje correctas |
| Precision | 0-1 (mayor mejor) | De las predichas, cuántas correctas |
| Recall | 0-1 (mayor mejor) | De las reales, cuántas detectadas |
| F1-Score | 0-1 (mayor mejor) | Balance P y R |
| AUC | 0.5-1 (mayor mejor) | Capacidad discriminativa |
| Loss | 0-∞ (menor mejor) | Error del modelo |
| Top-K Acc | 0-1 (mayor mejor) | Correcta en top K predicciones |

---

## 9. REFERENCIAS

- MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler et al., 2018)
- Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)
- Batch Normalization: Accelerating Deep Network Training (Ioffe & Szegedy, 2015)
- Deep Learning (Goodfellow, Bengio & Courville, 2016)