# Registro continuo — WanGP Safari

## Base
2026-10-23: repositorio auditado en4c87f4f; tres archivos originales, sin inferencia Wan. Autorización explícita de desarrollo autónomo y publicación solo enbryanstl1/wangp-ios. No modificar media kit ni otros repositorios.

## Trabajo implementado
- [x] Inventario completo y auditoría Wan/WanGP/Safari con fuentes primarias.
- [x] Presupuesto de pesos y activaciones (estimaciones distinguidas de medidas).
- [x] Lector safetensors robusto, HTTP206 y archivo local, límites, cancelación, F16/BF16.
- [x] Referencia CPU y bloques lineales WebGPU con liberación y detección de pérdida/errores.
- [x] Forward completo de un bloque T2V Wan, con pesos oficiales fijados, lectura por filas.
- [x] Oráculo PyTorch independiente y comparación real con27 tensores oficiales.
- [x] Interfaz móvil que separa funciones operativas y pendientes.
- [x] Caché opcional limitada, utilidades CPU INT4/INT8, informes JSON.
- [x] Pruebas automatizadas y definición de GitHub Actions.
- [ ] Generación de video local verificada en Safari iPhone (criterio final NO cumplido).

## Evidencia disponible
`docs/VERIFICATION.json`, `tests/fixtures/official-block-provenance.json`, `docs/REFERENCE.md`. Datos originales fuera de git en/tmp/wan-official, regenerables con scripts/fetch-official.py. Sin pesos de modelo en el repositorio.

## Decisiones de ingeniería
- Mantener producción sin dependencias JS externas y preservar laboratorio anterior.
- Verificar un bloque antes de extender el motor: una interfaz de generación sin pipeline sería engañosa.
- Batch1 y diagnóstico máximo64 tokens; la UI usa2. Atención CPU no sirve como backend eficiente para video grande.
- No anunciar INT4/GGUF/FP8 como generación compatible: los kernels experimentales y las utilidades CPU no equivalen a un pipeline cuantizado validado.
- No crear límite ficticio de RAM/temperatura: esas magnitudes no se exponen de forma portable.
- No activar inferencia remota ni adoptar un modelo alternativo silenciosamente.
- Plan original completo no ejecutable en esta entrega parcial: faltan subsistemas sustanciales; no marcar proyecto terminado.

## Próximos hitos con criterios concretos
1. Portar tokenizer SentencePiece/normalización + UMT5 encoder con pesos en formato seguro, fijar fixture de texto y error. Su embedding exige particionar tablas y evaluar cuantización con tolerancias.
2. Patch/time/text embeddings y denoiser completo; prueba de un paso contra PyTorch en geometría diminuta, antes de30 capas o resolución grande.
3. Scheduler flow-matching y guidance: latentes por paso contra referencia fijando semilla.
4. VAE temporal causal: Conv3D/resampling, atención, normas y caché temporal; validar reconstrucción de latentes contra referencia.
5. Encode/mux de frames generados a archivo reproducible. No validar con frames falsos como si procedieran del modelo.
6. Medir pico total y rendimiento del pipeline integrado en Safari iOS real, luego decidir configuración mínima o incompatibilidad concreta. Solo entonces video de aceptación reproducible.

## Reanudar sin repetir
Leer README, este registro, docs/AUDIT, la referencia y git log. Ejecutar npm ci, npm test y los tests indicados. Mantener botón Generar cerrado. Cualquier cambio de inferencia local a otra máquina requiere decisión explícita del usuario.

## Revisión de esta iteración
Revisión independiente parcial: detectó conservación de un objeto IndexedDB cerrado tras pagehide/BFCache. Se restablecen las referencias y se añadió regresión de ciclo de vida. El revisor no completó toda la revisión debido al límite de uso de su sesión; se continúa con revisión propia, sin afirmar una revisión independiente completa.

Endurecimiento posterior: manifiesto SHA256 por cada fragmento de128 filas de los27 tensores oficiales. La app verifica todos los fragmentos antes de ejecutar sus operadores; no se atribuye autenticidad a un archivo solo por su cabecera. Pruebas de GPU incluyen cancelación y pérdida del dispositivo con cero buffers vivos. Las claves de solicitudes Range a HF resolve incluyen el rango, manteniendo intactas URLs firmadas/personalizadas.

## Verificación previa a publicación
20 pruebas Node pasan; sintaxis de JS/inline HTML pasa; kernels WGSL compilados y ejecutados; bloque oficial CPU y WebGPU + prueba UI/lector íntegra pasan, alteración de peso rechazada; GPU device-loss/cancel con0 buffers vivos. UI móvil y caché/lifecycle pasan en Chromium y WebKit Linux (no iPhone). Se instalaron bibliotecas de WebKit solo en el entorno de pruebas; no se cambió la app para simular soporte.

## Experimento INT8/INT4 sobre pesos reales
Se implementaron kernels empaquetados con dequantización fusionada. El bloque completo coincide con CPU dequantizada (maxAbs5.01e-6/4.77e-6), pero difiere del original F32 (maxAbs0.01708 INT8 y0.30499 INT4). Pico de buffers GPU1,292,800/719,360 bytes, respectivamente. Se mantiene F32 por defecto y no se anuncia soporte de generación cuantizada. Evidencia y pérdidas publicadas en docs/QUANTIZATION.md y VERIFICATION.json.

## Publicación y prueba de red real
Se integró aa025d6 en main después de las comprobaciones de rama aprobadas. Pages build40003032878 y checks40003032617 aprobaron; se compararon14 archivos publicados con SHA256 exacto. La app pública leyó185,762,816 bytes desde HF y ejecutó el bloque en Chromium SwiftShader, sin transporte de pesos simulado: maxAbs7.51e-6,0 buffers vivos. No es una prueba de Safari iOS ni generación de video. Véanse docs/LIVE_VERIFICATION.json y DEPLOYMENT.json.

Después se agregó una ventana secuencial de lectura de hasta4MiB para reutilizar bytes entre los fragmentos GPU, reduciendo solicitudes repetidas; la app descarta esa referencia al terminar/cancelar. Pasan22 pruebas y nuevamente la integración completa de pesos oficiales con el lector/UI. Sin cambiar el número de tokens ni habilitar generación.

## Grabación de Safari recibida
El 23 de octubre se inspeccionó una grabación móvil de66,33s. Muestra la inspección compatible y la prueba de bloque en progreso hasta `ffn.2 · 1024/1536 filas · 160,7 MiB` en Safari. No muestra comparación final ni video; no marcar prueba completa de iPhone. Véase `docs/SAFARI-RECORDING.md`. La grabación reveló un informe viejo y el rótulo «WebGPU: Sin comprobar» durante la prueba; se vacía el informe al comenzar y se actualiza el estado de GPU durante y después del bloque. La integración automatizada comprueba esos estados.
