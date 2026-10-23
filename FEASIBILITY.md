# WanGP iPhone Safari — evaluación de viabilidad (2026-10-23)

## Objetivo invariable
Ejecutar generación de video con pesos reales Wan **dentro de Safari en iPhone 16 Pro/Pro Max**, con WebGPU local, sin inferencia remota ni app instalada.

## Estado verificado
- WebGPU compute, multiplicación 128×128 y atención sintética: pruebas manuales reportadas exitosas por el usuario.
- Cuantización INT8: demostración sobre 4096 valores sintéticos, ejecutada en CPU JavaScript.
- **No se han cargado pesos Wan reales ni ejecutado inferencia del modelo. No existe generación de video.**
- GitHub Pages aloja solo HTML/JS y no proporciona GPU ni Python.

## Dependencias reales
WanGP oficial (https://github.com/deepbeepmeep/WanGP) utiliza Python/PyTorch y kernels CUDA/ROCm. No es un backend ejecutable directamente en Safari.
Wan 2.1 T2V 1.3B (https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) tiene 30 capas Transformer, ancho 1536, T5 para texto y VAE temporal. La documentación de Diffusers (https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/wan.md) cita 8.19 GB de VRAM para inferencia de referencia.

## Presupuesto mínimo de pesos, sin activaciones ni buffers
1.3e9 parámetros: FP16 ~2.42 GiB; INT8 ~1.21 GiB; INT4 ~0.61 GiB. Estos valores **excluyen** T5, VAE, activaciones, latentes, almacenamiento intermedio y memoria de Safari. Cuantizar no implica soporte de kernels cuantizados.

## Criterios de salida para afirmar «funciona»
1. Descarga autenticada y comprobación de integridad de un checkpoint Wan oficial.
2. Lectura y conversión comprobada de sus tensores, con pruebas contra resultados de referencia.
3. Ejecución de una capa Transformer real con sus pesos en Safari.
4. Ejecución de todo el denoiser, codificador de texto y VAE con presupuesto de memoria medido en iPhone.
5. Generación y exportación de un video auténtico, repetible y verificable, completamente local.

## Riesgos bloqueantes
- Memoria disponible para Safari, límites de buffers WebGPU y cierres por presión de memoria.
- Port de operadores y cuantización compatible con WebGPU.
- Descarga/almacenamiento de varios GiB en iOS.
- Rendimiento térmico y duración de inferencia.

## Decisión técnica
**No declarar completado ni anunciar generación local hasta superar el criterio 5.**
La interfaz de diagnóstico actual no es un port funcional de WanGP.
