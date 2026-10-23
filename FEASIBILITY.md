# WanGP Safari — estado de viabilidad (2026-10-23)

El objetivo sigue siendo video IA local en Safari/iPhone, sin inferencia remota ni app adicional.

La evaluación anterior a4c87f4f se conserva en el historial. La auditoría actual está en [docs/AUDIT.md](docs/AUDIT.md), con fuentes, presupuestos de memoria y diferencias entre hechos medidos y estimaciones.

Se ha implementado y verificado numéricamente el bloque0 completo de Wan2.1 T2V1.3B con27 tensores oficiales y entradas artificiales. Esto supera las antiguas pruebas de matrices/atención/suma de cuadrados. CPU y Chromium WebGPU coinciden con referencia PyTorch dentro de tolerancia. [Registro](docs/VERIFICATION.json).

**No existe todavía una generación de video. No se ha ejecutado el bloque en Safari iOS/A18 Pro desde este entorno. No se ha probado una imposibilidad de hardware.**

Los bloqueos de software pendientes son UMT5/tokenizador, embeddings de entrada/tiempo/texto, denoiser completo, scheduler/denoising con guidance, VAE y exportación. También faltan kernels cuantizados y atención eficiente a escala de video. El botón Generar queda desactivado hasta completar y verificar esos componentes.

Para mantener todas las condiciones del usuario debe completarse y medirse ese port local. Un ordenador propio con GPU y acceso desde Safari sería una alternativa sin pagar API, pero movería la inferencia fuera del iPhone; no se ha usado ni sustituido silenciosamente el requisito original.
