#!/usr/bin/env python3
"""Independent CPU PyTorch reference for one Wan2.1 T2V attention block.

Synthetic default: python scripts/reference.py --output tests/fixtures/block-reference.json
Official weights: python scripts/reference.py --weights /tmp/wan-official --output /tmp/wan-real-reference.json
Requires torch. Does not import the JavaScript implementation or run a diffusion model.
"""
import argparse
import hashlib
import json
import math
import pathlib
import torch
import torch.nn.functional as F


def pattern(shape, seed, scale=0.1, offset=0.0):
    # Float64 sin followed by explicit float32 rounding; persisted values are authoritative.
    a = torch.arange(math.prod(shape), dtype=torch.float64)
    return (torch.sin((a + 1) * (seed + 1) * 0.173) * scale + offset).float().reshape(shape)


def make_weights(dim, hidden):
    shapes = {'modulation': [1, 6, dim], 'norm3.weight': [dim], 'norm3.bias': [dim],
              'ffn.0.weight': [hidden, dim], 'ffn.0.bias': [hidden],
              'ffn.2.weight': [dim, hidden], 'ffn.2.bias': [dim]}
    for kind in ['self_attn', 'cross_attn']:
        for projection in ['q', 'k', 'v', 'o']:
            shapes[f'{kind}.{projection}.weight'] = [dim, dim]
            shapes[f'{kind}.{projection}.bias'] = [dim]
        for norm in ['norm_q', 'norm_k']:
            shapes[f'{kind}.{norm}.weight'] = [dim]
    return {name: pattern(shape, i + 7, offset=1 if name.endswith(('norm_q.weight', 'norm_k.weight', 'norm3.weight')) else 0)
            for i, (name, shape) in enumerate(sorted(shapes.items()))}


def rope(x, grid):
    # WanModel concatenates independently generated temporal/height/width tables.
    length, heads, head_dim = x.shape
    pairs = head_dim // 2
    split = [pairs - 2 * (pairs // 3), pairs // 3, pairs // 3]
    base = torch.cat([10000 ** (-torch.arange(width, dtype=torch.float64) / width)
                      for width in split])
    angles = []
    for t in range(grid[0]):
        for h in range(grid[1]):
            for w in range(grid[2]):
                positions = torch.tensor([t] * split[0] + [h] * split[1] + [w] * split[2])
                angles.append(positions * base)
    multiplier = torch.polar(torch.ones((length, pairs), dtype=torch.float64), torch.stack(angles))
    complex_x = torch.view_as_complex(x.double().reshape(length, heads, pairs, 2))
    return torch.view_as_real(complex_x * multiplier[:, None, :]).flatten(2).float()


def forward(weights, x, context, e, config):
    dim, heads, eps = config['dim'], config['numHeads'], config['eps']
    def linear(a, prefix):
        return F.linear(a, weights[prefix + '.weight'], weights[prefix + '.bias'])
    def layernorm(a, affine=False):
        return F.layer_norm(a, [dim], weights['norm3.weight'] if affine else None,
                            weights['norm3.bias'] if affine else None, eps)
    def rms(a, name):
        return a * torch.rsqrt(a.square().mean(-1, keepdim=True) + eps) * weights[name + '.weight']
    def attention(a, source, kind, rotary=False):
        q = rms(linear(a, kind + '.q'), kind + '.norm_q').reshape(-1, heads, dim // heads)
        k = rms(linear(source, kind + '.k'), kind + '.norm_k').reshape(-1, heads, dim // heads)
        v = linear(source, kind + '.v').reshape(-1, heads, dim // heads)
        if rotary:
            q, k = rope(q, config['grid']), rope(k, config['grid'])
        scores = torch.einsum('lhd,shd->hls', q, k) / math.sqrt(dim // heads)
        attended = torch.einsum('hls,shd->lhd', scores.softmax(-1), v).reshape(-1, dim)
        return linear(attended, kind + '.o')
    modulation = (weights['modulation'].reshape(6, dim) + e).unbind(0)
    y = layernorm(x) * (1 + modulation[1]) + modulation[0]
    x = x + attention(y, y, 'self_attn', True) * modulation[2]
    x = x + attention(layernorm(x, True), context, 'cross_attn')
    y = layernorm(x) * (1 + modulation[4]) + modulation[3]
    y = linear(F.gelu(linear(y, 'ffn.0'), approximate='tanh'), 'ffn.2')
    return x + y * modulation[5]


def load_official(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    result = {}
    for name, spec in manifest['tensors'].items():
        data = (directory / spec['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != spec['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
        dtype = {'F32': torch.float32, 'BF16': torch.bfloat16, 'F16': torch.float16}[spec['dtype']]
        result[name.removeprefix('blocks.0.')] = torch.frombuffer(bytearray(data), dtype=dtype).reshape(spec['shape']).float()
    return result, {k: manifest[k] for k in ['model', 'revision', 'headerSha256', 'totalBytes']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    config = {'dim': 1536 if args.weights else 12, 'numHeads': 12 if args.weights else 2,
              'ffnDim': 8960 if args.weights else 20, 'eps': 1e-6, 'grid': [1, 1, 2]}
    dim = config['dim']
    weights, provenance = load_official(args.weights) if args.weights else (make_weights(dim, config['ffnDim']), {'kind': 'synthetic deterministic weights'})
    inputs = {'x': pattern([2, dim], 1, 0.3), 'context': pattern([3, dim], 2, 0.2), 'e': pattern([6, dim], 3, 0.05)}
    output = forward(weights, **inputs, config=config)
    record = {'schema': 'wan-block-reference-v1', 'config': config, 'provenance': provenance,
              'runtime': {'torch': torch.__version__, 'dtype': 'float32', 'device': 'cpu'},
              'inputs': {k: {'shape': list(v.shape), 'data': v.flatten().tolist()} for k, v in inputs.items()},
              'output': {'shape': list(output.shape), 'data': output.flatten().tolist()}}
    if not args.weights:
        record['weights'] = {k: {'shape': list(v.shape), 'data': v.flatten().tolist()} for k, v in weights.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Saved {args.output}; shape={list(output.shape)}, finite={bool(output.isfinite().all())}, sum={output.sum().item():.9g}')


if __name__ == '__main__':
    main()
