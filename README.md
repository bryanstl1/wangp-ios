# WanGP Safari

[Aplicación](https://bryanstl1.github.io/wangp-ios/) · [Auditoría](docs/AUDIT.md) · [Registro de avance](PROGRESS.md) · [Verificación](docs/VERIFICATION.json)

**Estado: motor parcial. Todavía NO genera video.** El objetivo sigue siendo generación local en Safari/iPhone, sin APIs de inferencia ni aplicaciones adicionales. No se ha probado el pipeline completo ni se ha demostrado que sea imposible en A18 Pro.

## Funciones reales

- Inspección de checkpoints `.safetensors` desde archivo local o HTTP Range/CORS. Rechaza offsets/shape incoherentes, solapamientos, huecos, claves duplicadas, cabeceras enormes y descargas completas accidentales.
- Conversión F32, F16 y BF16 a F32; lectura progresiva en fragmentos de hasta 4 MiB, máximo de salida de tensor de 64 MiB. No ejecuta archivos pickle `.pth`.
- Forward completo del **bloque 0 de Wan2.1 T2V 1.3B**: atención propia multihead con RMSNorm y RoPE 3D, atención cruzada con LayerNorm afín, FFN GELU tanh, modulación y residuales. Proyecciones en WebGPU; otras operaciones en JavaScript. 2 tokens, contexto3, activaciones artificiales conocidas. No usa el prompt ni ejecuta denoising.
- Carga de filas de pesos de 128 en 128; liberación de todos los buffers GPU tras cada proyección parcial. Se verificaron 27 tensores oficiales (185,762,816 bytes) mediante SHA256 en el test reproducible. La app también autentica cada fragmento de filas contra el manifiesto fijado antes de calcular.
- Comparación de la salida con referencia PyTorch independiente. CPU maxAbs `1.67e-6`; Chromium SwiftShader WebGPU maxAbs `7.51e-6`, tolerancia `2e-4 + 2e-4*abs(referencia)`. **No es una medición de Safari/iPhone.** Detalles en docs/VERIFICATION.json.
- Interfaz móvil, borrador local, progreso de filas realmente calculadas, cancelación entre fragmentos, cancelación al pasar a segundo plano y descarga del informe JSON.
- Caché IndexedDB opcional y limitada a 256 MiB para fragmentos oficiales, SHA256 por fragmento almacenado, evacuación de entradas antiguas y borrado. Safari puede eliminarla. La caché no es el modelo completo ni garantiza uso sin conexión.
- Kernels WebGPU INT8/INT4 empaquetados con dequantización fusionada, verificados en el bloque oficial frente a CPU dequantizada. Reducen sus buffers GPU, pero **no coinciden con la referencia F32 original**; siguen experimentales y no se habilitan como opciones de generación. [Resultados y límites](docs/QUANTIZATION.md). No hay pipeline de video cuantizado ni loader GGUF/FP8.

## Uso mínimo desde iPhone

1. Abre la aplicación en Safari con WebGPU disponible (WebKit anuncia soporte a partir de Safari26).
2. Pulsa **Comprobar WebGPU**. La app informa el resultado real, sin deducir RAM o temperatura a partir de un nombre de dispositivo.
3. En Pesos y motor, pulsa **Usar checkpoint oficial fijado**, luego **Inspeccionar modelo**. Se leen unos83 KiB de cabecera; el archivo completo tiene5.68GB.
4. Para ejecutar la capa, pulsa **Verificar bloque Wan**. La lectura de los pesos seleccionados es de unos186MB; usa Wi-Fi si quieres evitar consumir datos móviles. Mantén Safari en primer plano. Se puede cancelar.
5. Guarda el informe JSON si quieres conservar el resultado de ese dispositivo. **Generar video permanece desactivado.**

El servidor de modelos debe permitir CORS, exponer Content-Range y respetar HTTP206. Una red, un proxy o una CDN que devuelva otro fragmento provoca un error seguro; nunca se acepta una respuesta completa de gigabytes. Archivo local evita CORS, pero debe ser el checkpoint original compatible para la prueba del bloque.

## Pruebas reproducibles

```sh
npm ci
npx playwright install --with-deps chromium webkit
npm run check
npm test
npm run test:browser
node tests/gpu.mjs
```

`test:browser` valida UI móvil en Chromium y WebKit Linux, archivos corruptos, borrador y almacenamiento. WebKit Linux no sustituye Safari iOS/Metal. `gpu.mjs` compila y ejecuta WGSL y verifica el bloque sintético contra PyTorch en Chromium SwiftShader.

Para volver a obtener la referencia y verificar pesos oficiales, sin descargar el checkpoint entero:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/fetch-official.py --output /tmp/wan-official
python scripts/reference.py --weights /tmp/wan-official --output /tmp/wan-real-reference.json
node tests/official.mjs
```

Consulta [proveniencia y referencia](docs/REFERENCE.md). GitHub Actions reproduce unidades, JavaScript, navegadores, WGSL y el bloque oficial. Los pesos no se guardan en git ni se sirven desde Pages. Se mantiene la publicación de Pages existente desde `main`; el verificador de despliegue compara archivos publicados con el checkout.

## Pendiente para video real

Tokenización + UMT5-XXL, embeddings de texto/tiempo/patches, ejecución de las30 capas, scheduler y bucle de denoising con guidance, VAE temporal y codificación/mux de video. Deben verificarse extremo a extremo, optimizarse para el presupuesto real del teléfono y pasar una generación reproducible en Safari iOS. **No basta con que pase una capa.**

La auditoría calcula13.461GiB de pesos combinados en F16 o3.365GiB en INT4 ideal (sin escalas ni activaciones). No hay un límite de RAM de Safari comprobado aquí. La carga secuencial y atención por bloques requieren ingeniería adicional; no autorizan afirmar viabilidad ni imposibilidad.

## Desarrollo y licencias

ES modules sin dependencias en producción. `node scripts/serve.mjs` sirve la app en localhost:8000. El laboratorio original se conserva en `legacy-lab.html`; la ruta antigua `checkpoint-inspector.html` redirige al inspector actualizado. [Licencias y atribución](THIRD_PARTY_NOTICES.md). Este es un proyecto independiente, no una distribución oficial de WanGP.
