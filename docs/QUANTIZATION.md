# Cuantización experimental: precisión y memoria de una capa

Se implementaron kernels WebGPU con pesos agrupados INT8 e INT4 empaquetados en palabras u32, extracción con signo y dequantización fusionada dentro de la multiplicación. Escalas por64 valores y activaciones/bias/acumulación F32. El código no requiere tipos nativos WGSL i8/i4.

Los27 tensores oficiales del bloque0 se autentican antes de calcular. Cada fragmento F32 se cuantiza en CPU sobre la marcha; no es un checkpoint cuantizado distribuido ni un loader GGUF/FP8. No reduce el tráfico de descarga de este experimento. CPU y GPU aún necesitan temporales F32, y la caché conserva fragmentos F32.

Prueba: bloque Wan2.1 T2V1.3B completo,2 tokens y3 tokens de contexto artificial; mismo fixture que la referencia F32. Chromium SwiftShader, **no iPhone**. Las cifras son la suma de buffers WebGPU gestionados simultáneamente, no toda la memoria del navegador.

| Método | Pico de buffers GPU | Error máximo frente a CPU dequantizada | Error máximo frente a PyTorch F32 original | Coincide con referencia F32 a2e-4+2e-4*r |
|---|---:|---:|---:|---|
| F32 | 4,661,760 bytes | — | 0.00000751 | Sí |
| INT8 | 1,292,800 bytes | 0.00000501 | 0.0170801 | **No** |
| INT4 | 719,360 bytes | 0.00000477 | 0.304989 | **No** |

Los kernels empaquetados coinciden con una referencia CPU que dequantiza los mismos pesos. Eso verifica extracción, escalas y multiplicación; no prueba fidelidad al modelo original. La diferencia frente a F32 ya es visible en una sola capa y puede propagarse por30 capas y varios pasos. No se ha medido calidad de video ni se puede inferirla de este fixture artificial.

Por ese motivo la aplicación mantiene **F32 como única ruta de verificación por defecto**. INT4/INT8 son diagnósticos de ingeniería disponibles en la API del motor y las pruebas, no opciones de generación anunciadas como compatibles. Se necesita evaluación de esquemas por canal, excepciones para outliers, calibración y/o quantization-aware training, además de la implementación y validación del pipeline entero.

Reproducir: comandos oficiales en README, luego `node tests/official.mjs`. El informe `quantizedDiagnostics` conserva tanto las comparaciones aprobadas del operador como las comparaciones fallidas frente al original. No se ocultan esas pérdidas ni se presenta cuantización como garantía de viabilidad.
