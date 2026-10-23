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
- No anunciar INT4/GGUF/FP8 como compatibles: las utilidades de cuantización CPU no equivalen a kernels del modelo.
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
