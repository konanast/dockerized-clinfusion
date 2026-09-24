#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path

BASE = Path('/opt/ClinFusion/cache/models')

SHARED_MODELS = {
    'dinov2': {
        'repo': 'facebook/dinov2-large',
        'dir': 'dinov2-large',
        'required': ['config.json', 'preprocessor_config.json'],
        'weight_any': ['model.safetensors', 'pytorch_model.bin'],
    },
    'convnext': {
        'repo': 'laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup',
        'dir': 'CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup',
        'required': ['open_clip_config.json', 'open_clip_pytorch_model.bin'],
        'weight_any': [],
    },
}

VARIANT_MODELS = {
    '8B': {
        'qwen': {
            'repo': 'Qwen/Qwen3-VL-8B-Instruct',
            'dir': 'Qwen3-VL-8B-Instruct',
            'required': ['config.json', 'preprocessor_config.json', 'tokenizer_config.json'],
            'weight_any': ['model.safetensors.index.json', 'model.safetensors'],
        },
        'clinfusion': {
            'repo': 'Alibaba-DAMO-Academy/ClinFusion-8B',
            'dir': 'ClinFusion-8B',
            'required': ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'model.safetensors.index.json'],
            'weight_any': [],
        },
    },
    '32B': {
        'qwen': {
            'repo': 'Qwen/Qwen3-VL-32B-Instruct',
            'dir': 'Qwen3-VL-32B-Instruct',
            'required': ['config.json', 'preprocessor_config.json', 'tokenizer_config.json'],
            'weight_any': ['model.safetensors.index.json', 'model.safetensors'],
        },
        'clinfusion': {
            'repo': 'Alibaba-DAMO-Academy/ClinFusion-32B',
            'dir': 'ClinFusion-32B',
            'required': ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'model.safetensors.index.json'],
            'weight_any': [],
        },
    },
}


def normalize_variant(value=None):
    raw = str(value or os.getenv('MODEL_VARIANT', '8B')).strip().upper().replace('-', '')
    aliases = {'8': '8B', '8B': '8B', '32': '32B', '32B': '32B'}
    if raw not in aliases:
        raise ValueError(f"Unsupported MODEL_VARIANT={value!r}; use 8B or 32B")
    return aliases[raw]


def models_for_variant(variant=None):
    variant = normalize_variant(variant)
    # Preserve the familiar qwen/dinov2/convnext/clinfusion display order.
    return {
        'qwen': VARIANT_MODELS[variant]['qwen'],
        'dinov2': SHARED_MODELS['dinov2'],
        'convnext': SHARED_MODELS['convnext'],
        'clinfusion': VARIANT_MODELS[variant]['clinfusion'],
    }


def _index_missing(path: Path):
    idx = path / 'model.safetensors.index.json'
    if not idx.exists():
        return []
    try:
        data = json.loads(idx.read_text())
        shards = sorted(set(data.get('weight_map', {}).values()))
        return [name for name in shards if not (path / name).is_file() or (path / name).stat().st_size == 0]
    except Exception as e:
        return [f'invalid model.safetensors.index.json ({e})']


def inspect_one(name, spec):
    path = BASE / spec['dir']
    missing = []
    if not path.is_dir():
        return {'name': name, 'repo': spec['repo'], 'path': str(path), 'ready': False, 'missing': ['directory']}
    for f in spec['required']:
        p = path / f
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(f)
    if spec['weight_any'] and not any((path / f).is_file() and (path / f).stat().st_size > 0 for f in spec['weight_any']):
        missing.append('one of: ' + ', '.join(spec['weight_any']))
    missing.extend(_index_missing(path))
    return {'name': name, 'repo': spec['repo'], 'path': str(path), 'ready': not missing, 'missing': missing}


def inspect_all(variant=None):
    variant = normalize_variant(variant)
    return {name: inspect_one(name, spec) for name, spec in models_for_variant(variant).items()}


def print_status(status, variant):
    print(f'\nClinFusion {variant} model file status')
    print('=' * 72)
    for name, s in status.items():
        mark = 'OK' if s['ready'] else 'MISSING/INCOMPLETE'
        print(f"{name:10s} {mark:18s} {s['path']}")
        if s['missing']:
            shown = s['missing'][:8]
            for m in shown:
                print(f'             - {m}')
            if len(s['missing']) > len(shown):
                print(f"             - ... {len(s['missing'])-len(shown)} more")
    print()


def download(names, variant):
    specs = models_for_variant(variant)
    BASE.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
    for name in names:
        spec = specs[name]
        target = BASE / spec['dir']
        target.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading/resuming {spec['repo']} -> {target}")
        cmd = ['hf', 'download', spec['repo'], '--local-dir', str(target)]
        subprocess.run(cmd, check=True, env=env)


def main():
    p = argparse.ArgumentParser(description='Check and resume ClinFusion 8B/32B model downloads')
    p.add_argument('--variant', default=os.getenv('MODEL_VARIANT', '8B'), help='8B or 32B (default: MODEL_VARIANT or 8B)')
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('--prompt', action='store_true')
    d = sub.add_parser('download')
    d.add_argument('models', nargs='*', choices=['qwen', 'dinov2', 'convnext', 'clinfusion'], default=[])
    sub.add_parser('json')
    args = p.parse_args()

    variant = normalize_variant(args.variant)
    status = inspect_all(variant)
    if args.cmd == 'json':
        print(json.dumps({'variant': variant, 'models': status}, indent=2))
        return
    if args.cmd == 'download':
        names = args.models or list(models_for_variant(variant))
        download(names, variant)
        status = inspect_all(variant)
        print_status(status, variant)
        raise SystemExit(0 if all(x['ready'] for x in status.values()) else 2)

    print_status(status, variant)
    incomplete = [n for n, s in status.items() if not s['ready']]
    if not incomplete:
        print(f'All required ClinFusion {variant} model files look complete.')
        return
    print('Incomplete:', ', '.join(incomplete))
    if args.prompt and os.isatty(0):
        ans = input(f'Download/resume the incomplete {variant} models now? [y/N] ').strip().lower()
        if ans in ('y', 'yes'):
            download(incomplete, variant)
            status = inspect_all(variant)
            print_status(status, variant)
            raise SystemExit(0 if all(x['ready'] for x in status.values()) else 2)
        print('Skipped. The API can still start, but selected-model loading remains unavailable until files are complete.')
    raise SystemExit(2)


if __name__ == '__main__':
    main()
