# Attribution and licenses

The new ES-module runtime and CPU reference implement the architecture documented by [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1), Copyright 2024-2025 The Alibaba Wan Team Authors. Wan2.1 code and the selected official Wan-AI/Wan2.1-T2V-1.3B model are published under Apache-2.0. The equations, tensor naming, modulation order and RoPE structure are adapted to browser JavaScript; this is an independent, incomplete browser implementation. The original Apache license text is in LICENSES/Wan2.1-Apache-2.0.txt. Checkpoint weights are downloaded from their upstream publisher only on request and are not bundled here. The small numerical fixtures record our tests of this architecture.

[deepbeepmeep/WanGP](https://github.com/deepbeepmeep/WanGP) is a project reference only. Its current additional implementation is under the WanGP Community License 2.0, which is distinct from Apache-2.0. No WanGP-specific Python, memory-management, UI, or productization code is bundled or ported. This repository is not an official WanGP release and has no affiliation or endorsement.

Development-only dependencies are separately installed, not shipped to the browser: Playwright/@playwright/test (Apache-2.0), PyTorch (BSD-3-Clause), GitHub Actions (licenses in their upstream repositories). The production app uses browser APIs and no CDN scripts, paid API or inference service. The host and upstream model-download servers receive normal static-file requests. Prompts and activations are not submitted for remote inference.

Existing laboratory experiments from commit 4c87f4f are retained in legacy-lab.html and checkpoint-inspector.html. Original authorship remains unchanged.
