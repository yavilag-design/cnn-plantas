# RESUMEN EJECUTIVO

## Clasificación de Enfermedades en Plantas mediante Visión por Computador y Deep Learning

---

### El Problema

Las enfermedades en cultivos representan una amenaza crítica para la seguridad alimentaria global, causando pérdidas que pueden superar el 30% de la producción mundial de alimentos. Los métodos tradicionales de diagnóstico dependen de inspectores expertos, siendo procesos lentos, costosos y sujetos a variabilidad humana.

### La Solución

Desarrollamos un sistema inteligente basado en visión por computadora y deep learning que permite clasificar automáticamente enfermedades en plantas a partir de imágenes digitales de hojas, utilizando técnicas de transfer learning con la arquitectura MobileNetV2.

---

### Resultados Clave

| Métrica | Valor |
|---------|-------|
| Precisión de validación | **95.02%** |
| AUC-ROC | **99.84%** |
| Precisión Top-3 | **99.43%** |
| Tiempo de entrenamiento | ~2 horas |
| Clases reconocidas | 38 |

---

### Características Técnicas

- **Dataset:** PlantVillage con 54,305 imágenes de 14 cultivos
- **Arquitectura:** MobileNetV2 pre-entrenada + capas personalizadas
- **Parámetros:** 2.6M totales (366K entrenables)
- **Preprocesamiento:** Filtro gaussiano 5x5 para reducción de ruido
- **Optimización:** Adam con early stopping y reducción automática de learning rate

---

### Aplicaciones

- Monitoreo de cultivos por drones
- Aplicaciones móviles para agricultores
- Sistemas de control de calidad industrial
- Investigación agrícola y epidemiología vegetal

---

### Impacto Potencial

Detección temprana de enfermedades → Menor uso de fungicidas → Producción sostenible → Seguridad alimentaria global

---

### Diferenciadores

- Transfer learning eficiente que reduce tiempo de entrenamiento
- Arquitectura optimizada para dispositivos edge y móviles
- Alta capacidad discriminativa (AUC-ROC 99.84%)
- Listo para despliegue en campo
