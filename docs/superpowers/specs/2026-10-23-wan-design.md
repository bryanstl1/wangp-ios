# Local Wan Safari engineering scope

User specification: real local video in Safari iPhone A18 Pro, no paid services, no extra app, modify only wangp-ios. Explicit authorization to execute/publish without intermediate questions.

Preserve existing experiments as legacy-lab.html. Introduce strict random-access safetensors reader (remote HTTP 206 and local File), F32/F16/BF16 conversion, bounded reads, cancellation, tensor validation. Implement batch-one Wan2.1 T2V attention block with original naming, streamed linear weights, CPU reference and WebGPU linear kernel. Use explicit synthetic activations for layer diagnostics; never call this video inference. Compare against independent Python/PyTorch and real pinned official weights when network permits.

Approaches considered: port Python/CUDA runtime (not directly executable in browser); assemble unverified complete pipeline (would misrepresent functionality); incremental verified browser-native runtime (chosen engineering decision). Full T5 tokenizer/encoder, patch/time/text embedding path, complete denoising scheduler, temporal VAE and export remain acceptance gates, not presumed implemented. A complete block alone must not enable Generate.

UI: mobile first, prompt/settings retained locally, operative diagnostics separated from missing pipeline stages. Display measured capability limits, unknown process memory/temperature. No inference endpoints. Cache only explicit bounded reads; no full multi-GB eager download. Keep tests and source provenance in repository.
