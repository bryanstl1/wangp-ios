# Wan2.1 single-block numerical reference

This validates a batch-one T2V transformer attention block. Inputs are artificial
activations, projected text context, and timestep modulation. It does **not** run
the text encoder, VAE, scheduler, diffusion loop, or produce video.

## Reproduce

Python requires CPU PyTorch (tested with 2.14.1+cpu). The downloader uses only the
Python standard library.

```sh
python scripts/reference.py --output tests/fixtures/block-reference.json
python scripts/fetch-official.py --output /tmp/wan-official
python scripts/reference.py --weights /tmp/wan-official --output /tmp/wan-real-reference.json
```

`block-reference.json` contains complete deterministic synthetic weights, inputs,
and float32 expected output: dimension 12, two heads, FFN width 20, two tokens,
three context tokens, and grid `[1,1,2]`. Every tensor is `{shape, data}` with
row-major flattened data and PyTorch linear weights in `[out,in]` order. This
small fixture exercises every block operation without distributing model weights.

The official run uses dimension 1536, 12 heads, FFN width 8960, and the same token
counts. Its weights remain outside the repository. The output JSON contains
inputs and expected outputs but no weights. `official-block-provenance.json`
contains source offsets, tensor shapes/dtypes, and SHA256 checksums for all 27
downloaded tensors.

## Provenance and transport bounds

Checkpoint: [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/37ec512624d61f7aa208f7ea8140a131f93afc9a),
revision `37ec512624d61f7aa208f7ea8140a131f93afc9a`, original
`diffusion_pytorch_model.safetensors`. Only names beginning `blocks.0.` are
downloaded: 185,762,816 bytes total, all F32. The entire checkpoint is
5,676,070,424 bytes and is never downloaded by this script.

Each request has a Range header, a distinct range query key, a 16 MiB response
bound, strict HTTP 206/Content-Range validation, and an exact byte-count check.
The header is limited to 4 MiB, selected block to 256 MiB, and concurrency to four
requests. Failed downloads leave `.part` files and do not write a success
manifest. Tensor checksums are verified again by the reference reader.

Original raw safetensors header SHA256:
`2ec9c33e0081ab18b8faadfc85efb5d370b74778000ac1c2180c9b66613ef0f0`.

The independent reference follows the [official model source](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py).
The inspected source file SHA256 was
`c1572ade3bf7345bd4c00d4a19535788fab62f9c08b5aeb15f6d01f4435bb47a`.
RMSNorm spans the full model dimension before head splitting. Self attention uses
three independently generated RoPE frequency tables, concatenated in temporal,
height, width order. Cross attention has affine LayerNorm; FFN uses tanh GELU.
The six modulation channels are shift, scale, gate for self attention, then
shift, scale, gate for the FFN. LayerNorm and RMSNorm epsilon is 1e-6.

## Recorded verification

Both synthetic and official-weight references were compared against upstream `WanAttentionBlock`,
`WanSelfAttention`, `WanT2VCrossAttention`, normalization and RoPE implementations,
with flash attention replaced by a CPU float32 softmax attention expression.
Maximum absolute difference was **0**. This checks the independent reference's
architecture; it does not assert equality to a particular GPU flash kernel.

Both synthetic and official runs produced finite outputs. The official output
shape is `[2,1536]`, sum `12.2465286`. Float32 reductions can vary slightly by
runtime; compare with absolute and relative tolerances rather than JSON hashes.
The generated official reference JSON SHA256 for the recorded environment was
`c7777106a6a197a97d9137bbf53b97acfdf77c5a0482bf0a68b332fd26a31a38`.
