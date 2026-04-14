# PRESENTACIÓN EJECUTIVA

## Clasificación de Enfermedades en Plantas mediante Visión por Computador

**Visión por Computador - Actividad 6**

---

## Diapositiva 1: Título

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║       CLASIFICACIÓN DE ENFERMEDADES EN PLANTAS                   ║
║       MEDIANTE VISION POR COMPUTADOR Y DEEP LEARNING             ║
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │                                                          │    ║
║  │    🍎  🫐  🍇  🍅  🌽  🥔  🍓                            │    ║
║  │    ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓                 │    ║
║  │                                                          │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                                                                  ║
║                    Accuracy: 95.02%                              ║
║                    AUC-ROC: 99.84%                               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Diapositiva 2: El Problema

```
┌─────────────────────────────────────────────────────────────────┐
│                    EL DESAFÍO AGRÍCOLA                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🌾 Perdidas por enfermedades en cultivos: >30%               │
│                                                                 │
│  ❌ Diagnóstico tradicional:                                    │
│     • Lento (días/semanas para resultados)                     │
│     • Costoso (personal especializado)                           │
│     • Sujeto a error humano                                     │
│                                                                 │
│  ✅ Solución: Visión por Computador                            │
│     • Análisis instantáneo de imágenes                          │
│     • Escalable y reproducible                                   │
│     • Disponible para cualquier agricultor                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 3: Dataset PlantVillage

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATASET UTILIZADO                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 Estadísticas del Dataset                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  54,305 imágenes totales                                 │   │
│  │  ├── 43,456 para entrenamiento (80%)                     │   │
│  │  └── 10,849 para validación (20%)                        │   │
│  │  38 clases (14 cultivos × estados de salud)              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  🍎 Manzana (4)    🍇 Uva (4)       🍅 Tomate (10)            │
│  🫐 Arándano (1)   🌽 Maíz (4)      🥔 Papa (3)               │
│  🍒 Cereza (2)     🍊 Naranja (1)   🍓 Fresa (2)             │
│  🍑 Durazno (2)    🫑 Pimiento (2)  🥬 Soja (1)               │
│  🫐 Frambuesa (1)  🎃 Calabaza (1)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 4: Arquitectura del Modelo

```
┌─────────────────────────────────────────────────────────────────┐
│              ARQUITECTURA: MOBILENETV2 + TRANSFER LEARNING      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IMAGEN                          ARQUITECTURA                   │
│  224×224×3                       ─────────────                  │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────┐                                           │
│  │  MobileNetV2    │  ◄── Pre-entrenada con ImageNet          │
│  │  (congelada)    │      2.2M parámetros fijos               │
│  │                 │                                           │
│  │  Depthwise      │                                           │
│  │  Separable Conv │                                           │
│  └────────┬────────┘                                           │
│           │ 7×7×1280                                            │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Global Average  │  ◄── 7×7 → 1×1 (reduce parámetros)       │
│  │ Pooling         │                                           │
│  └────────┬────────┘                                           │
│           │ 1280                                                 │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Dropout (30%)   │  ◄── Regularización                       │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Dense(256)+ReLU │                                           │
│  │ BatchNorm       │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Dropout (30%)   │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Dense(128)+ReLU │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Dense(38)+Soft  │  ◄── 38 probabilidades (clases)         │
│  └─────────────────┘                                           │
│                                                                 │
│  Total parámetros: 2.6M | Entrenables: 366K                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 5: Preprocesamiento - Filtro Gaussiano

```
┌─────────────────────────────────────────────────────────────────┐
│           PREPROCESAMIENTO: FILTRO GAUSSIANO                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ¿QUÉ ES?                                                       │
│  ──────────                                                     │
│  Un filtro de suavizado que usa una distribución gaussiana      │
│  para ponderar los píxeles vecinos durante la convolución.     │
│                                                                 │
│  FUNDAMENTO MATEMÁTICO:                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │    G(x,y) = (1/2πσ²) × e^(-(x²+y²)/2σ²)                 │   │
│  │                                                          │   │
│  │    σ = desviación estándar                               │   │
│  │    x,y = distancia al centro del kernel                  │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  KERNEL GAUSSIANO 5×5:                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │     1    4    7    4    1     ← Píxeles más cercanos     │   │
│  │     4   16   26   16    4         tienen mayor peso      │   │
│  │     7   26   41   26    7     ← Centro: peso máximo     │   │
│  │     4   16   26   16    4                                 │   │
│  │     1    4    7    4    1                                 │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 6: Justificación Técnica del Filtro Gaussiano

```
┌─────────────────────────────────────────────────────────────────┐
│         ¿POR QUÉ FILTRO GAUSSIANO? - JUSTIFICACIÓN TÉCNICA      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PROBLEMA: RUIDO EN IMÁGENES DE CAMPO                           │
│  ───────────────────────────────────────                        │
│  • Ruido de sensor en condiciones de poca luz                   │
│  • Compresión JPEG introduce artifacts                          │
│  • Variaciones de iluminación naturales                          │
│                                                                 │
│  ┌──────────────────┬──────────────────────────────────────┐   │
│  │    SIN FILTRO    │         CON FILTRO GAUSSIANO         │   │
│  ├──────────────────┼──────────────────────────────────────┤   │
│  │                  │                                        │   │
│  │ ░▓░▓░▓░▓░▓░▓░▓  │        ░░▒▒▓▓▓▓▒▒░░░░                 │   │
│  │ ▓░▓░▓░▓░▓░▓░▓░▓  │        ░░▒▒▓▓▓▓▒▒░░░░                 │   │
│  │ ░▓░▓░▓░▓░▓░▓░▓  │        ░░▒▒▓▓▓▓▒▒░░░░                 │   │
│  │                  │                                        │   │
│  │ Ruido de alta    │        Suavizado preserva              │   │
│  │ frecuencia        │        estructuras principales         │   │
│  │                  │                                        │   │
│  └──────────────────┴──────────────────────────────────────┘   │
│                                                                 │
│  VENTAJAS DEL FILTRO GAUSSIANO vs OTRAS ALTERNATIVAS:           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Característica      │ Gaussiano │ Box Blur │ Mediana   │   │
│  │ ───────────────────────────────────────────────────────│   │
│  │ Preserva bordes      │    ✓✓✓    │    ✓     │    ✓✓    │   │
│  │ Elimina ruido        │    ✓✓✓    │    ✓✓    │    ✓✓✓   │   │
│  │ Sin artifacts        │    ✓✓✓    │    ✗     │    ✓✓    │   │
│  │ Isotrópico           │    ✓✓✓    │    ✓✓    │    ✗     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ELECCIÓN DEL KERNEL 5×5:                                        │
│  ─────────────────────────                                       │
│  • 3×3 → Suavizado insuficiente                                 │
│  • 5×5 → EQUILIBRIO ÓPTIMO ◄──                                 │
│  • 7×7 → Riesgo de perder detalles de enfermedad                │
│  • 9×9 → Demasiado difuso                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 7: Resultados - Métricas Principales

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESULTADOS DEL MODELO                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║                                                           ║  │
│  ║          ┌──────────────────────────────────────┐         ║  │
│  ║          │                                      │         ║  │
│  ║          │       PRECISIÓN DE VALIDACIÓN       │         ║  │
│  ║          │                                      │         ║  │
│  ║          │              95.02%                   │         ║  │
│  ║          │                                      │         ║  │
│  ║          │         ██████████████████░░░░       │         ║  │
│  ║          │                                      │         ║  │
│  ║          └──────────────────────────────────────┘         ║  │
│  ║                                                           ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
│                                                                 │
│  MÉTRICAS COMPARADAS:                                           │
│  ┌────────────────────────┬────────────┬────────────┐         │
│  │ Métrica                │ Entreno    │ Validación │         │
│  ├────────────────────────┼────────────┼────────────┤         │
│  │ Accuracy               │  92.84%    │  95.02%    │         │
│  │ Precision              │  94.34%    │  95.84%    │         │
│  │ Recall                 │  91.78%    │  94.39%    │         │
│  │ AUC-ROC                │  99.80%    │  99.84%    │         │
│  │ Top-3 Accuracy         │  99.13%    │  99.43%    │         │
│  │ Loss                   │   0.207    │   0.148    │         │
│  └────────────────────────┴────────────┴────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 8: Curvas de Aprendizaje

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURVAS DE APRENDIZAJE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ACCURACY vs ÉPOCA                    LOSS vs ÉPOCA            │
│  ┌─────────────────────────┐          ┌─────────────────────────┐│
│  │                         │          │                         ││
│  │    ┌────────────────┐  │          │    ┌────────────────┐  ││
│  │  1 │  ╲             │  │          │    │                ╱  ││
│  │    │   ╲            │  │          │    │               ╱   ││
│  │  0.9│    ╲           │  │          │  0.3│              ╱    ││
│  │    │     ╲ Train     │  │          │    │             ╱     ││
│  │  0.8│      ╲─ ─ ─ ─ ─│  │          │  0.2│           ╱      ││
│  │    │        ╲ Val     │  │          │    │         ╱        ││
│  │  0.7│         ╲──────│  │          │  0.1│       ╱         ││
│  │    │  1  2  3  4  5  │  │          │    │     ╱           ││
│  │    └────────────────┘  │          │    │   ╱             ││
│  │              Épocas    │          │    │  ╱ Train         ││
│  └─────────────────────────┘          │    │ ╱ Val           ││
│                                        │    │1  2  3  4  5   ││
│  INTERPRETACIÓN:                       │    └────────────────┘│
│  ────────────────                       └─────────────────────────┘│
│  ✓ Val > Train → Buen generalization                              │
│  ✓ Curvas convergentes → Sin overfitting                          │
│  ✓ Pérdida decreciendo → Aprendizaje efectivo                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 9: Rendimiento por Clase

```
┌─────────────────────────────────────────────────────────────────┐
│                   RENDIMIENTO POR CLASE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CLASES CON MEJOR RENDIMIENTO (F1-Score):                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  Corn_healthy          ████████████████████████████ 1.00│   │
│  │  Orange_Haunglongbing   ████████████████████████████ 1.00│   │
│  │  Grape_Leaf_blight      ████████████████████████████ 0.99│   │
│  │  Strawberry_Leaf_scorch  ████████████████████████████ 0.99│   │
│  │  Soybean_healthy         ████████████████████████████ 0.99│   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  CLASES CON MENOR RENDIMIENTO:                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  Potato_healthy     ███████████████░░░░░░░░░░░░░░░░  0.74│   │
│  │  Tomato_Early_blight ███████████████░░░░░░░░░░░░░░░░  0.72│   │
│  │  Tomato_Target_Spot  ████████████████░░░░░░░░░░░░░░  0.79│   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  CAUSAS DE ERROR:                                               │
│  • Desbalance de clases (Potato_healthy solo 30 samples)        │
│  • Similitud visual entre enfermedades del tomate                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 10: Tiempo y Recursos

```
┌─────────────────────────────────────────────────────────────────┐
│                  TIEMPO Y RECURSOS COMPUTACIONALES              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CONFIGURACIÓN DE ENTRENAMIENTO:                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  • Optimizador:        Adam (lr=0.001)                   │   │
│  │  • Batch size:         32 imágenes                       │   │
│  │  • Épocas:             10 (óptimo: 9 por early stop)     │   │
│  │  • Pasos por época:    1,358 batches                     │   │
│  │  • Callbacks:          EarlyStopping + ReduceLROnPlateau  │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  TIEMPO DE ENTRENAMIENTO:                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │              ⏱ 7,450 segundos                            │   │
│  │              ⏱ ~2.07 horas                              │   │
│  │              ⏱ ~12.4 minutos por época                   │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  MEDIDAS DE OPTIMIZACIÓN:                                       │
│  • EarlyStopping: Evita entrenamiento innecesario               │
│  • ReduceLROnPlateau: Fine-tuning automático                    │
│  • ModelCheckpoint: Guarda mejor modelo automáticamente          │
│  • GPU no disponible: Entrenamiento en CPU (~2 horas)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 11: Comparación con Estado del Arte

```
┌─────────────────────────────────────────────────────────────────┐
│              COMPARACIÓN CON ESTADO DEL ARTE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PRECISIÓN EN PLANTVILLAGE:                                     │
│                                                                 │
│  100%┤                                                          │
│      │                    ┌────────┐                             │
│   99%┤                    │ Dense  │  99.7%                      │
│      │                    │Net161  │                             │
│   98%┤                    └────────┘                             │
│      │                                                          │
│   97%┤          ┌────────┐                                       │
│      │          │AlexNet│  97.7%                                 │
│   96%┤          └────────┘                                       │
│      │                                                          │
│   95%┤┌────────┐                                               │
│      ││Mobile  │  95.0%                                         │
│   94%┤│NetV2   │  ◄── NUESTRO MODELO                           │
│      │└────────┘                                               │
│   93%┤                                                          │
│      └───────────────────────────────────────────────────       │
│           AlexNet     MobileNetV2     DenseNet161                │
│           (2016)        (2026)        (2019)                    │
│                                                                 │
│  ANÁLISIS:                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Modelo        │ Precisión │ Ventaja                      │   │
│  │ ──────────────────────────────────────────────────────  │   │
│  │  DenseNet161   │  99.7%    │ Mayor precisión              │   │
│  │  MobileNetV2   │  95.0%    │ ✓ Más eficiente (menor)      │   │
│  │                │           │ ✓ Ideal para dispositivos     │   │
│  │                │           │   edge/móviles               │   │
│  │                │           │ ✓ Deployment accesible        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 12: Aplicaciones

```
┌─────────────────────────────────────────────────────────────────┐
│                    APLICACIONES DEL MODELO                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🚁 MONITOREO POR DRONES                                         │
│     ╔═══════════════════════════════════════════╗               │
│     ║  Drones con cámaras multiespectrales      ║               │
│     ║  Análisis en tiempo real de cultivos       ║               │
│     ║  Detección temprana de focos de enfermedad║               │
│     ╚═══════════════════════════════════════════╝               │
│                                                                 │
│  📱 APLICACIONES MÓVILES                                       │
│     ╔═══════════════════════════════════════════╗               │
│     ║  App para agricultores                    ║               │
│     ║  Diagnóstico instantáneo desde el campo  ║               │
│     ║  Consejos de tratamiento automatizados    ║               │
│     ╚═══════════════════════════════════════════╝               │
│                                                                 │
│  🏭 SISTEMAS INDUSTRIALES                                       │
│     ╔═══════════════════════════════════════════╗               │
│     ║  Control de calidad en empaque            ║               │
│     ║  Clasificación automatizada de productos  ║               │
│     ║  Trazabilidad fitosanitaria               ║               │
│     ╚═══════════════════════════════════════════╝               │
│                                                                 │
│  🔬 INVESTIGACIÓN AGRÍCOLA                                     │
│     ╔═══════════════════════════════════════════╗               │
│     ║  Estudios epidemiológicos                 ║               │
│     ║  Monitoreo de resistencia a fungicidas    ║               │
│     ║  Validación de variedades resistentes    ║               │
│     ╚═══════════════════════════════════════════╝               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 13: Limitaciones y Mejoras Futuras

```
┌─────────────────────────────────────────────────────────────────┐
│               LIMITACIONES Y TRABAJO FUTURO                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LIMITACIONES ACTUALES:                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  ❌ Imágenes de fondo controlado (hojas recortadas)       │   │
│  │  ❌ Solo análisis de hoja, no planta completa              │   │
│  │  ❌ Clases desbalanceadas                                 │   │
│  │  ❌ Condiciones de iluminación no variadas                │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  MEJORAS PROPUESTAS:                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  ✅ Data augmentation avanzado (rotaciones, perspectivas)│   │
│  │  ✅ Dataset con fondos naturales y condiciones de campo   │   │
│  │  ✅ Ensemble de modelos (combinar arquitecturas)          │   │
│  │  ✅ Segmentaciónsemántica (detectar región afectada)      │   │
│  │  ✅ Despliegue en dispositivos edge (Raspberry Pi)        │   │
│  │  ✅ Transfer learning con datasets de campo específicos    │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 14: Conclusiones

```
┌─────────────────────────────────────────────────────────────────┐
│                         CONCLUSIONES                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║                                                           ║  │
│  ║   ✅ Modelo exitoso de clasificación de enfermedades      ║  │
│  ║                                                           ║  │
│  ║   ✅ 95.02% precisión en 38 clases                        ║  │
│  ║                                                           ║  │
│  ║   ✅ Transfer learning efectivo con MobileNetV2           ║  │
│  ║                                                           ║  │
│  ║   ✅ Filtro gaussiano mejora robustez del modelo          ║  │
│  ║                                                           ║  │
│  ║   ✅ AUC-ROC de 99.84% indica excelente discriminación    ║  │
│  ║                                                           ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
│                                                                 │
│  APRENDIZAJES CLAVE:                                            │
│  ────────────────────                                           │
│  • Transfer learning acelera desarrollo y mejora resultados     │
│  • Preprocesamiento adecuado reduce sensibilidad a ruido      │
│  • Regularización (dropout, early stopping) previene overfitting │
│  • Balance entre precisión y eficiencia es crucial para deploy   │
│                                                                 │
│  IMPACTO POTENCIAL:                                             │
│  ──────────────────                                             │
│  Detección temprana → Menor uso de fungicidas →                 │
│  Producción sostenible → Seguridad alimentaria global            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 15: Referencias

```
┌─────────────────────────────────────────────────────────────────┐
│                        REFERENCIAS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [1] Abdallah, A., Ali, A., & Ramdan, M. (2023). PlantVillage   │
│      Dataset. Kaggle.                                            │
│                                                                 │
│  [2] Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016).       │
│      Using deep learning for image-based plant disease          │
│      detection. Frontiers in Plant Science, 7, 1419.             │
│                                                                 │
│  [3] Too, E. C., et al. (2019). A comparative study of          │
│      fine-tuning deep learning models for plant disease         │
│      identification. Computers and Electronics in Agriculture.   │
│                                                                 │
│  [4] Sandler, M., et al. (2018). MobileNetV2: Inverted          │
│      residuals and linear bottlenecks. CVPR.                    │
│                                                                 │
│  [5] Rosebrock, A. (2019). Deep Learning for Computer Vision     │
│      with Python. Apress.                                        │
│                                                                 │
│  [6] Sharda, R., et al. (2021). Analytics, Data Science,        │
│      Artificial Intelligence. Pearson Educación.                │
│                                                                 │
│  [7] Contreras-Bravo, L., et al. (2024). Algoritmos             │
│      supervisados y de ensamble con Python. Ediciones de la U.  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diapositiva 16: Gracias

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║                                                                  ║
║                     ¡GRACIAS POR SU ATENCIÓN!                    ║
║                                                                  ║
║                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │                                                          │   ║
║  │     "La tecnología es mejor cuando conecta a las         │   ║
║  │      personas con la tierra que las alimenta"           │   ║
║  │                                                          │   ║
║  └──────────────────────────────────────────────────────────┘   ║
║                                                                  ║
║                      ¿PREGUNTAS?                                 ║
║                                                                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

*Presentación generada para el curso de Visión por Computador - Actividad 6*
*Duración estimada: 5-7 minutos de presentación*
