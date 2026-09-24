import gc
import os
import threading
import time

import torch
from PIL import Image
from scripts.model_manager import inspect_all, normalize_variant


class ModelRuntime:
    def __init__(self):
        self.adapter = None
        self.loading = False
        self.load_error = None
        self._lock = threading.RLock()
        self.variant = normalize_variant(os.getenv('MODEL_VARIANT', '8B'))
        self.model_path = f'cache/models/ClinFusion-{self.variant}'
        self.model_id = f'clinfusion-{self.variant.lower()}'

    def file_status(self):
        return inspect_all(self.variant)

    def files_ready(self):
        return all(v['ready'] for v in self.file_status().values())

    def gpu_status(self):
        out = {
            'available': bool(torch.cuda.is_available()),
            'torch': torch.__version__,
            'hip': torch.version.hip,
        }
        if torch.cuda.is_available():
            out['name'] = torch.cuda.get_device_name(0)
            try:
                free_b, total_b = torch.cuda.mem_get_info(0)
                out['memory_free_gib'] = round(free_b / 2**30, 2)
                out['memory_total_gib'] = round(total_b / 2**30, 2)
            except Exception:
                pass
        return out

    def memory_advisory(self):
        if self.variant != '32B':
            return None
        total = self.gpu_status().get('memory_total_gib')
        if total is not None and total < 90:
            return (
                f'32B selected but PyTorch reports only {total} GiB GPU-visible memory. '
                'On a 128-GB Strix Halo, increase the host TTM/GTT allocation (commonly around 100 GiB) before loading 32B.'
            )
        return '32B selected. Keep batch_size/concurrency at 1 and leave host RAM headroom; 8B remains the recommended default.'

    def status(self):
        return {
            'model': self.model_id,
            'variant': self.variant,
            'model_path': self.model_path,
            'loaded': self.adapter is not None,
            'loading': self.loading,
            'load_error': self.load_error,
            'files_ready': self.files_ready(),
            'models': self.file_status(),
            'gpu': self.gpu_status(),
            'memory_advisory': self.memory_advisory(),
            'dtype': os.getenv('CLINFUSION_DTYPE', 'fp16'),
            'attention': os.getenv('CLINFUSION_ATTN_IMPL', 'sdpa'),
        }

    def load(self):
        with self._lock:
            if self.adapter is not None:
                return self.status()
            if not self.files_ready():
                raise RuntimeError(
                    f'Required {self.variant} model files are incomplete. '
                    f'Run ./models.sh {self.variant} or docker compose --profile tools run --rm setup --variant {self.variant} check --prompt'
                )
            if not torch.cuda.is_available():
                raise RuntimeError('ROCm GPU is not visible to PyTorch inside the container')
            self.loading = True
            self.load_error = None
            try:
                os.chdir('/opt/ClinFusion')
                from custom_model.medevalkit_adapter_qwen3_vl import MedEvalKitAdapter
                self.adapter = MedEvalKitAdapter(
                    model_path=self.model_path,
                    model_config={},
                    generation_config={
                        'temperature': 0.0,
                        'top_p': 1.0,
                        'repetition_penalty': 1.0,
                        'max_new_tokens': int(os.getenv('DEFAULT_MAX_NEW_TOKENS', '128')),
                    },
                )
                return self.status()
            except Exception as e:
                self.adapter = None
                self.load_error = f'{type(e).__name__}: {e}'
                raise
            finally:
                self.loading = False

    def unload(self):
        with self._lock:
            self.adapter = None
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return self.status()

    def infer(self, prompt, images=None, niftis=None, max_new_tokens=128, temperature=0.0, top_p=1.0, repetition_penalty=1.0):
        if self.adapter is None:
            self.load()
        images = images or []
        niftis = niftis or []
        with self._lock:
            adapter = self.adapter
            old = (adapter.max_new_tokens, adapter.temperature, adapter.top_p, adapter.repetition_penalty)
            adapter.max_new_tokens = int(max_new_tokens)
            adapter.temperature = float(temperature)
            adapter.top_p = float(top_p)
            adapter.repetition_penalty = float(repetition_penalty)
            try:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
                t0 = time.perf_counter()
                batch = {'messages': {'prompt': prompt}}
                opened_images = []
                if images:
                    for x in images:
                        with Image.open(str(x)) as im:
                            opened_images.append(im.convert('RGB'))
                    batch['messages']['image'] = opened_images
                if niftis:
                    batch['messages']['nifti'] = [str(x) for x in niftis]
                output = adapter.generate([batch])[0]
                elapsed = time.perf_counter() - t0
                try:
                    out_tokens = len(adapter.processor.tokenizer.encode(output, add_special_tokens=False))
                except Exception:
                    out_tokens = None
                peak = None
                try:
                    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
                except Exception:
                    pass
                return {
                    'output': output,
                    'usage': {'completion_tokens': out_tokens},
                    'metrics': {
                        'variant': self.variant,
                        'total_seconds': round(elapsed, 3),
                        'effective_output_tokens_per_second': round(out_tokens / elapsed, 3) if out_tokens is not None and elapsed else None,
                        'peak_torch_allocated_gib': peak,
                        'note': 'Effective rate includes image/volume preprocessing and prefill; it is not decode-only tokens/sec.',
                    },
                }
            finally:
                adapter.max_new_tokens, adapter.temperature, adapter.top_p, adapter.repetition_penalty = old


runtime = ModelRuntime()
