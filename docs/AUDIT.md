# Auditoría: Wan 2.1 T2V 1.3B en Safari de iPhone

Fecha: 2026-10-23. Objetivo conservado: inferencia con pesos Wan reales, enteramente local en Safari de iPhone 16 Pro/Pro Max, sin servidor de inferencia ni app instalada. Las referencias [S1]–[S12] están en [SOURCES.md](SOURCES.md).

## Resultado y alcance

**Este repositorio no contiene inferencia completa de Wan ni generación de video.** La auditoría de partida en4c87f4f leyó sus tres archivos originales completos: `index.html`, `checkpoint-inspector.html` y `FEASIBILITY.md`. El laboratorio implementa operaciones sintéticas pequeñas; el inspector puede leer metadatos safetensors y aplicar una suma de cuadrados a un tensor pequeño. Ninguna de esas operaciones ejecuta una capa Wan. Los éxitos manuales del usuario son evidencia de esos diagnósticos en su dispositivo, no una medición de Wan. Esta auditoría no ejecutó Safari en un iPhone.

No se ha demostrado que todo port posible sea imposible. Tampoco hay evidencia suficiente para prometer que un port completo, útil y estable quepa o termine en ese teléfono. Tras la implementación de esta sesión ya existe un bloque completo con proyecciones WebGPU, lectura por filas e integridad por fragmento; su comparación numérica con PyTorch pasó con pesos oficiales. Véase [VERIFICATION.json](VERIFICATION.json). Faltan el runtime integral y su medición en iPhone. Un HTML estático de GitHub Pages puede distribuir una implementación, pero no aporta Python, PyTorch, CUDA ni GPU remota.

## Arquitectura comprobada

WanGP es una aplicación Python con interfaz web y dependencias de inferencia nativas; abrir su interfaz desde Safari no mueve la inferencia al teléfono. Su documentación/dependencias no constituyen un backend WebGPU [S6]. La arquitectura objetivo aquí es Wan **2.1 T2V 1.3B**, no las variantes 14B, I2V ni Wan 2.2.

| Componente | Configuración oficial |
|---|---|
| Denoiser DiT | 30 bloques; ancho **1536**; FFN **8960**; **12** cabezas, dimensión por cabeza 128 |
| Video | 16 canales latentes; compresión VAE temporal/espacial **(4,8,8)**; parche DiT **(1,2,2)** |
| Texto | **UMT5-XXL encoder**, tokenizer `google/umt5-xxl`, máximo **512** tokens; salida de ancho **4096**, proyectada a 1536 |
| UMT5 encoder | vocabulario 256384; ancho 4096; FFN 10240; 24 bloques; 64 cabezas; posición relativa por bloque |
| Atención DiT | autoatención global, cross-attention al texto; normalización Q/K y de cross-attention |
| VAE | autoencoder causal 3D; ancho base 96, z=16, convoluciones, normalización, atención y cachés temporales |

Estos datos proceden de las configuraciones y módulos oficiales [S1–S4]. El pipeline oficial codifica prompt positivo y negativo, crea ruido latente, repite denoising con guidance y scheduler, y decodifica con VAE. Su referencia habitual usa 81 cuadros y 50 pasos T2V: con guidance son dos llamadas al denoiser por paso [S5]. Reducir resolución/cuadros no reduce el número de parámetros. Un caso diminuto sirve para validar ejecución, pero su calidad y utilidad fuera de la distribución habitual deben medirse.

## Archivos reales y presupuesto de pesos

El API oficial de archivos de Hugging Face devolvió estos tamaños exactos; no se descargaron los checkpoints completos [S7]. GB decimal y GiB binario no son intercambiables.

| Archivo original | Bytes de archivo | GiB | Formato observado |
|---|---:|---:|---|
| `diffusion_pytorch_model.safetensors` | 5,676,070,424 | 5.2863 | F32 en todos sus tensores de cabecera |
| `models_t5_umt5-xxl-enc-bf16.pth` | 11,361,920,418 | 10.5816 | PyTorch BF16 según checkpoint/configuración |
| `Wan2.1_VAE.pth` | 507,609,880 | 0.4727 | PyTorch FloatStorage según ficha oficial |
| **Total de estos tres archivos** | **17,545,600,722** | **16.3406** | excluye tokenizer y otros archivos |

Se solicitaron con HTTP Range únicamente los primeros 8 bytes y la cabecera de **83,216 bytes** del denoiser. Ambas respuestas fueron HTTP 206. La suma de formas da **1,418,996,800 parámetros F32** y 5,675,987,200 bytes de payload. El nombre comercial «1.3B» no es una cuenta exacta. Esto verifica metadatos del archivo, no integridad de todo el checkpoint ni ejecución del modelo.

UMT5 tiene **5,680,910,336** parámetros por suma de las capas oficiales: `256384*4096 + 24*(4*4096² + 3*4096*10240 + 2*4096 + 32*64) + 4096`. El VAE completo tiene **126,892,531**, contados instanciando su definición oficial en el dispositivo `meta` de PyTorch, sin cargar pesos: encoder 53,595,872; decoder 73,295,331; resto 1,328. Estos dos conteos son de arquitectura; no se inspeccionaron todos los almacenamientos de los `.pth`.

| Pesos sin activaciones, GiB | F32 (4 B/p) | F16/BF16 (2 B/p) | I8 ideal (1 B/p) | I4 ideal (0.5 B/p) |
|---|---:|---:|---:|---:|
| DiT 1,418,996,800 | 5.2862 | 2.6431 | 1.3215 | 0.6608 |
| UMT5 encoder 5,680,910,336 | 21.1630 | 10.5815 | 5.2908 | 2.6454 |
| VAE completo 126,892,531 | 0.4727 | 0.2364 | 0.1182 | 0.0591 |
| **Suma 7,226,799,667** | **26.9219** | **13.4610** | **6.7305** | **3.3652** |

Fórmula: `parámetros * bytes_por_parámetro / 2^30`. I8/I4 son **límites inferiores aritméticos**, no checkpoints disponibles o formatos ya compatibles: faltan escalas, ceros, alineación, metadatos, parámetros que permanezcan F32/F16 y buffers de dequantización. Cuantizar UMT5/VAE también exige validar error y calidad. Convertir todo a F16 no preserva automáticamente el rango de BF16. Estos totales no son RAM pico ni requisito de residencia simultánea: pueden descargarse componentes entre fases. Para T2V solo se necesita decodificar; retirar el encoder VAE puede ahorrar parte de sus pesos, pero requiere una exportación correcta y no elimina UMT5.

## Activaciones: diminuto frente a 480p

Ejemplo analítico, batch 1, sin padding adicional y con dimensiones divisibles por 16. `Tlat=(F-1)/4+1`, `N=Tlat*(H/16)*(W/16)`. Los datos siguientes son tamaños de **tensores individuales**, no mediciones ni sumas de pico. F32 duplica cada tamaño marcado F16.

| Tensor o forma | 5 cuadros, 64×64 | 81 cuadros, 480×832 |
|---|---:|---:|
| Tlat; tokens N | 2; **32** | 21; **32,760** |
| Latente F32 `16*Tlat*(H/8)*(W/8)` | 8 KiB | 7.998 MiB |
| Hidden F16 `N*1536` | 0.09375 MiB | 95.977 MiB |
| Q+K+V F16 `3*N*1536` | 0.28125 MiB | 287.930 MiB |
| Intermedio FFN F16 `N*8960` | 0.546875 MiB | 559.863 MiB |
| Scores de autoatención F16 `12*N²` | 24 KiB | **23.988 GiB** |
| Scores cross-attention F16 `12*N*512` | 0.375 MiB | 383.906 MiB |
| Video RGB F32 `3*F*H*W` | 0.234375 MiB | 370.195 MiB |

La matriz densa N² sería prohibitiva como estrategia práctica, pero **no es un requisito inevitable del modelo**: atención por tiles con softmax en línea evita materializarla completa. Aun así conserva trabajo de atención cuadrático y necesita validación. Procesar guidance secuencialmente evita duplicar batch; intermedios se reutilizan entre las 30 capas, no se multiplican sin motivo por 30. VAE con tiles/chunks puede reducir activaciones, pero necesita halos y caché causal correctos; no equivale a decodificar cuadros independientes. El cálculo excluye residuales coexistentes, acumulación F32, workspace de matmul, temporales del scheduler, staging CPU/GPU, tokenizer, medios de salida y memoria del navegador.

## Safari/WebGPU: límites reales y trabajo pendiente

WebKit anunció WebGPU en Safari 26.0 también para iOS [S8]. Eso justifica probar capacidades, no inferir compatibilidad de un modelo. El estándar diferencia límites de adaptador y de dispositivo, por ejemplo `maxBufferSize` y `maxStorageBufferBindingSize`; `requestDevice()` sin límites explícitos no solicita automáticamente todos los máximos del adaptador [S9]. Se deben registrar ambos. Los valores estándar de referencia (256 MiB de buffer y 128 MiB de binding de storage) **no son un cupo de RAM total de Safari**. La disponibilidad del dispositivo, presión de memoria, pérdidas de dispositivo, tamaño de almacenamiento y estabilidad deben medirse en cada versión de iOS/Safari y teléfono; no existe aquí evidencia de un límite universal fijo de 1, 2 o 4 GiB.

| Bloqueo de esta implementación | Lo que haría falta |
|---|---|
| Python/PyTorch y kernels CUDA/ROCm no corren directamente en una página | runtime JavaScript/Wasm + kernels WGSL o un runtime compatible con todos los operadores |
| No hay tokenizer UMT5 ni encoder ejecutable | tokenizer equivalente, embeddings, posición relativa, atención sin escalado T5, FFN gated, normalizaciones y pruebas contra referencia |
| Bloque0 implementado y validado; denoiser integral ausente | embeddings de entrada/texto/tiempo,30 bloques, cabeza de salida y validación de un paso completo; atención eficiente para secuencias grandes |
| Dtypes nativos incompatibles con una traducción literal | WGSL no ofrece BF16, complex128 ni f64 del código RoPE original; reescritura con pares reales/F32 y tolerancias comprobadas [S10] |
| I8/I4 no equivalen a matmul cuantizado | almacenamiento empaquetado, unpack/dequant y matmul fusionado; WGSL base no tiene un tipo escalar aritmético i4/i8 ni la API de PyTorch |
| VAE y sampler ausentes | convolución causal 3D, resample, atención, cachés, escalado latente, scheduler y RNG reproducible |
| Descarga, GPU y exportación sin ciclo integral | formato convertido/shardeado, integridad, gestión explícita de buffers, video y prueba repetible local |

`shader-f16` es una capacidad opcional que debe detectarse y solicitarse; `enable f16;` por sí solo no concede soporte [S9–S10]. Estos son bloqueos del código actual, no una prueba de que los operadores matemáticos sean inimplementables en WebGPU.

## Streaming y próximos experimentos

Range, shards y almacenamiento persistente pueden evitar tener el archivo y una copia convertida completos en el heap JS. Exigen CORS y respuestas Range correctas; almacenar en disco no aumenta RAM. Transmitir capas reduce pesos residentes y obliga a volver a leer/transferirlos en cada pasada. Con 50 pasos y dos pasadas de guidance hay hasta 100 recorridos del DiT: sus 0.6608 GiB I4 ideales implican unos **66.1 GiB** de tráfico de pesos si se releen íntegros cada vez, antes de escalas y sin prometer ninguna latencia. Reusar una capa para ambas ramas puede reducir lecturas a cambio de mantener dos flujos de activaciones. UMT5 se usa al inicio y el VAE al final; desalojarlos entre fases cambia el pico, sin borrar su coste de descarga/ejecución. Guardar embeddings de prompts precalculados limita el prompt; calcularlos en un servidor cambia el requisito local.

El hito **una capa real** con pesos/versiones identificados y comparación numérica con PyTorch ya se verificó en CPU y Chromium SwiftShader, con2 tokens artificiales y27 tensores oficiales. No es evidencia de Safari iOS ni de video. Los siguientes hitos son: DiT diminuto completo; UMT5 y VAE individualmente; pipeline completo con prompt/semilla, pico de memoria observable, tiempos, pérdidas de dispositivo y video exportado; finalmente aumento gradual hacia 480p. Un solo tensor, una cabecera o un video de demostración externo no sustituyen esos hitos. El pipeline integral todavía no está completado.

## Opciones gratuitas sin confundir el objetivo

| Opción | Dónde infiere | Relación con el objetivo |
|---|---|---|
| Continuar este port experimental | Safari local | conserva todas las restricciones; coste de desarrollo y viabilidad pendientes |
| Wan/WanGP en un ordenador compatible ya disponible | otro equipo local del usuario | software abierto; no es inferencia en iPhone ni ausencia de instalación [S1,S6] |
| Notebook de Google Colab con GPU gratuita cuando se asigne | **remoto** | alternativa de experimento; cuotas/GPU no garantizadas, no cumple el objetivo local [S11] |
| Hugging Face Space de terceros con cupo gratuito | **remoto** | depende del Space, cola, licencia y disponibilidad; el hosting gratuito base no garantiza GPU [S12] |

No se activó inferencia remota ni se rebajaron las restricciones locales. «Gratis» no demuestra capacidad del teléfono ni disponibilidad permanente de una GPU.
