# Grabación recibida de Safari — 2026-10-23

La persona usuaria compartió una grabación de pantalla de 66,33 s de WanGP Safari 0.7.0. SHA-256 del archivo original: `87a7a070e34942a89c23006ecb1f6180924342e5578e44f78545e6395a2b4b40`. El video y el texto del prompt no se incluyen en el repositorio.

## Lo observable

- La página móvil de Safari muestra una inspección del checkpoint fijado: 825 tensores, 5,68 GB de archivo y cabecera compatible. El informe visible conserva `videoGeneration: false` y `status: checkpoint-inspected`.
- En otra parte de la grabación se ve el avance de **Verificar bloque Wan**: `self_attn.o`, `cross_attn.q`, `ffn.0` y `ffn.2`. En torno a 52 s alcanza `ffn.2 · 1024/1536 filas · 160,7 MiB leídos (red o caché)`.
- Durante esa ejecución la ficha «Estado del dispositivo» todavía decía «WebGPU: Sin comprobar», porque solo se actualizaba al pulsar el diagnóstico separado. Además, el informe de la inspección previa seguía visible mientras se ejecutaba el bloque. Se corrigieron ambas señales de la interfaz.
- La grabación termina con la página desplazada hacia arriba. **No muestra** el estado final `official-block-verified`, la comparación frente a PyTorch, un informe descargado ni un video generado. No demuestra que el bloque completo haya terminado en el dispositivo o que el teléfono sea un modelo concreto.

## Consecuencia técnica

La grabación aporta evidencia de ejecución y lectura progresiva en Safari móvil, más allá de las pruebas automatizadas de WebKit Linux. Todavía se necesita el informe final de una ejecución en ese teléfono para atribuirle la verificación numérica completa. Las pruebas de escritorio y CI sí terminaron el bloque completo, pero no sustituyen esa evidencia de Safari iOS. La generación de video sigue deshabilitada por falta del codificador de texto, el denoiser completo, el VAE y la exportación.
