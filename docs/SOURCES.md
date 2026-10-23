# Fuentes primarias y reproducción de la auditoría

Consulta: 2026-10-23. Las ramas `main` son mutables. Se consultaron páginas oficiales con búsqueda web y se recuperaron código/configuraciones y metadatos por HTTPS desde CLI. No se descargaron los archivos de pesos completos. Los tamaños y conteos son una instantánea, no una promesa sobre futuras revisiones.

| ID | URL primaria | Evidencia usada |
|---|---|---|
| S1 | https://github.com/Wan-Video/Wan2.1 ; https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/wan_t2v_1_3B.py ; https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/shared_config.py | código de referencia, configuración exacta 1.3B, text_len 512, BF16 |
| S2 | https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py | patch embedding, 16 canales, texto4096, bloques, RoPE, norma y FFN |
| S3 | https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/t5.py | UMT5-XXL encoder, definición y conteo de parámetros, tokenizer y dtype |
| S4 | https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py | definición VAE, z16, base96, cachés temporales y parámetros meta |
| S5 | https://github.com/Wan-Video/Wan2.1/blob/main/wan/text2video.py ; https://github.com/Wan-Video/Wan2.1/blob/main/generate.py | geometría latente, guidance doble, fases, scheduler, 81 cuadros y pasos |
| S6 | https://github.com/deepbeepmeep/WanGP ; https://github.com/deepbeepmeep/WanGP/blob/main/requirements.txt | WanGP es software Python de inferencia con interfaz web; dependencias y hardware nativos |
| S7 | https://huggingface.co/api/models/Wan-AI/Wan2.1-T2V-1.3B/tree/main?recursive=true&expand=false ; https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main ; https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/diffusion_pytorch_model.safetensors | tamaños exactos, hashes LFS y cabecera del denoiser; la página de árbol mostraba revisión `37ec512` |
| S8 | https://webkit.org/blog/17333/webkit-features-in-safari-26-0/ | anuncio WebKit de WebGPU para Safari26.0/iOS; no prescribe RAM para modelos |
| S9 | https://gpuweb.github.io/gpuweb/ ; https://www.w3.org/TR/webgpu/ | límites de adaptador/dispositivo, requiredLimits, shader-f16, pérdida de dispositivo; no presupuesto fijo de memoria de Safari |
| S10 | https://gpuweb.github.io/gpuweb/wgsl/ ; https://www.w3.org/TR/WGSL/ | tipos numéricos WGSL y extensión f16; reescrituras necesarias frente a PyTorch |
| S11 | https://research.google.com/colaboratory/faq.html | recursos gratuitos dinámicos, límites y GPU no garantizados; alternativa remota |
| S12 | https://huggingface.co/docs/hub/spaces-overview ; https://huggingface.co/docs/hub/spaces-gpus | Spaces: hosting gratuito CPU y opciones GPU; no garantía general de video gratis ni ejecución en iPhone |

## Observaciones verificadas por CLI

1. La lista S7 devolvió 5,676,070,424; 11,361,920,418; 507,609,880 bytes para denoiser, UMT5 y VAE respectivamente.
2. Se enviaron solicitudes `Range: bytes=0-7` y `Range: bytes=8-83223` al denoiser. Se observó `206`, `Content-Range: bytes 0-7/5676070424` y `bytes 8-83223/5676070424`. La cabecera JSON declara 1,418,996,800 parámetros, todos F32; se sumó el producto de cada `shape` excluyendo `__metadata__`.
3. La definición S3 permite calcular el encoder sin cargar 11.36GB: 5,680,910,336 parámetros. Incluye embedding grande y posición relativa por cada bloque (`shared_pos=False`). No se contó el decoder de lenguaje, que no usa este pipeline.
4. Se recuperó S4 y se instanció únicamente su definición `WanVAE_` dentro de `torch.device('meta')`, con la configuración usada por `_video_vae` y z16. `sum(p.numel() for p in model.parameters())` dio 126,892,531. Sin tensores reales ni pesos descargados. La dependencia `einops.rearrange` no se invocó en ese conteo de constructores.
5. La aritmética de memoria y activaciones de AUDIT.md es cálculo explícito, no benchmark ni medición de Safari.

Hashes de contenido LFS publicados por S7 (SHA-256; **no verificados mediante descarga completa**):

| Archivo | SHA-256 esperado |
|---|---|
| denoiser | `96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f` |
| UMT5 | `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d` |
| VAE | `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` |

El paper oficial https://arxiv.org/abs/2503.20314 y README oficial citan 8.19GB de VRAM para inferencia de referencia. No se usa esa cifra como memoria mínima universal, ni como medición de iPhone: depende de runtime, precisión y offloading. Tener un pico de VRAM de escritorio menor que el total de los archivos es compatible con no tener todos los componentes residentes a la vez.
