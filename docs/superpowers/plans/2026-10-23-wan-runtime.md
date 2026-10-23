# Wan browser runtime implementation plan

> Execute continuously using superpowers:executing-plans. User waived approval checkpoints.

Goal: verified incremental local Wan operators and truthful mobile app; do not claim complete video.
Architecture: ES modules; random access checkpoint provider; streamed block linear; WebGPU compute; JS reference; Python independent oracle.
Tech: vanilla JS/WGSL, Node test runner, Playwright, GitHub Actions/Pages.
Spec: docs/superpowers/specs/2026-10-23-wan-design.md

Global constraints: only wangp-ios; no remote inference/costs; no fabricated outputs; preserve useful experiments.
Review focus: malformed/overlapping tensors; ignored Range; cancellation and device loss; stale UI async work; real layer versus synthetic fixtures.

- [x] Task 1: tests/checkpoint.test.js -> src/checkpoint.js. Validate dtype/shape/offset products, coverage, HTTP bounded reads, local Blob, F32/F16/BF16. RED then GREEN.
- [x] Task 2: tests/math.test.js + independent fixture -> src/math.js, src/wan-block.js, src/gpu.js. Batch1 block input/output and strict shape checks. GPU linear X@W.T+b with row chunks and lifetime cleanup. Compare CPU/GPU/PyTorch with explicit tolerances. RED then GREEN.
- [x] Task 3: tests/browser.mjs -> index.html, src/app.js, src/style.css. Operational file/URL inspector, diagnostics/block runner, cancel, report; disabled video controls with reasons, preserve legacy lab. RED then GREEN.
- [x] Task 4: .github/workflows/checks.yml, docs/AUDIT.md, README, PROGRESS.md. CI Node/Python/browser, pinned real-weight opt-in check, deployment after green checks. Run all checks, review, publish and verify deployed exact files.

Interfaces: Checkpoint.open(source,{signal,onProgress}), .tensors Map, .read(name,{elementOffset,count}), .summary(); provider get(name), linear(name,x,rows,inDim,outDim). runWanBlock({x,context,e,grid,config,provider,linear,signal,onProgress}) -> Float32Array.
