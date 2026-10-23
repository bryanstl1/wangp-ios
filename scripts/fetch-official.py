#!/usr/bin/env python3
"""Fetch only block zero from the pinned official checkpoint; no ML dependencies."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import re
import struct
import urllib.request

REVISION = '37ec512624d61f7aa208f7ea8140a131f93afc9a'
URL = f'https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/{REVISION}/diffusion_pytorch_model.safetensors'
MAX_HEADER = 4 * 1024 * 1024
MAX_BLOCK = 256 * 1024 * 1024
CHUNK = 16 * 1024 * 1024


def read_range(start, end):
    expected = end - start + 1
    if expected <= 0 or expected > CHUNK:
        raise ValueError('Unbounded range refused')
    # Distinct cache keys avoid proxies reusing a response for another Range.
    request = urllib.request.Request(f'{URL}?range={start}-{end}', headers={
        'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(request, timeout=120) as response:
        actual = response.headers.get('Content-Range', '')
        if response.status != 206 or not re.fullmatch(rf'bytes {start}-{end}/\d+', actual):
            raise RuntimeError(f'Server did not honor requested range: {response.status}, {actual}')
        if int(response.headers.get('Content-Length', expected)) != expected:
            raise RuntimeError('Unexpected Content-Length')
        data = response.read(expected + 1)
        if len(data) != expected:
            raise RuntimeError('Unexpected body length')
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/tmp/wan-official', type=pathlib.Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    length = struct.unpack('<Q', read_range(0, 7))[0]
    if not 2 < length <= MAX_HEADER:
        raise RuntimeError('Safetensors header exceeds bound')
    raw_header = read_range(8, 7 + length)
    header = json.loads(raw_header)
    selected = {name: spec for name, spec in header.items() if name.startswith('blocks.0.')}
    total = sum(v['data_offsets'][1] - v['data_offsets'][0] for v in selected.values())
    if not selected or total > MAX_BLOCK:
        raise RuntimeError(f'Invalid block selection ({total} bytes)')
    (args.output / 'header.json').write_text(json.dumps(header, indent=2) + '\n')
    def fetch(item):
        name, spec = item
        lo, hi = spec['data_offsets']
        target = args.output / (name + '.bin')
        digest = hashlib.sha256()
        temporary = target.with_suffix('.part')
        with temporary.open('wb') as stream:
            for offset in range(lo, hi, CHUNK):
                data = read_range(8 + length + offset, 8 + length + min(offset + CHUNK, hi) - 1)
                stream.write(data)
                digest.update(data)
        temporary.replace(target)
        print(f'{name}: {hi-lo} bytes', flush=True)
        return name, {**spec, 'file': target.name, 'byteLength': hi-lo, 'sha256': digest.hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        tensors = dict(executor.map(fetch, sorted(selected.items())))
    manifest = {'model': 'Wan-AI/Wan2.1-T2V-1.3B', 'revision': REVISION,
                'url': URL, 'headerSha256': hashlib.sha256(raw_header).hexdigest(),
                'totalBytes': total, 'tensors': tensors}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(tensors)} tensors, {total} bytes in {args.output}')


if __name__ == '__main__':
    main()
