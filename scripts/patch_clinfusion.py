#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/ClinFusion")
adapter = root / "custom_model" / "medevalkit_adapter_qwen3_vl.py"
openclip = root / "custom_model" / "openclip_encoder.py"

text = adapter.read_text()
old = 'torch_dtype="auto",\n            device_map={"": self.device},\n            low_cpu_mem_usage=True,\n            attn_implementation="flash_attention_2"'
new = 'torch_dtype={"fp16": torch.float16, "bf16": torch.bfloat16, "auto": "auto"}.get(os.environ.get("CLINFUSION_DTYPE", "fp16").lower(), torch.float16),\n            device_map={"": self.device},\n            low_cpu_mem_usage=True,\n            attn_implementation=os.environ.get("CLINFUSION_ATTN_IMPL", "sdpa")'
if old in text:
    text = text.replace(old, new, 1)
elif 'attn_implementation=os.environ.get("CLINFUSION_ATTN_IMPL", "sdpa")' not in text:
    raise SystemExit("Could not locate expected ClinFusion attention block; upstream changed.")


# Upstream selects the 32B base with a case-sensitive substring check even though
# the official directory is ClinFusion-32B. Make model-size selection robust.
old_selector = '}["32b" in model_path]'
new_selector = '}["32b" in model_path.lower()]'
if old_selector in text:
    text = text.replace(old_selector, new_selector, 1)
elif new_selector not in text:
    raise SystemExit("Could not locate expected ClinFusion 8B/32B model selector; upstream changed.")

# Upstream silently substitutes a zero volume after 3D preprocessing failure.
# For an API deployment this is unsafe: fail the request instead by default.
needle = """        except Exception as e:\n            print(f"Preprocessing failed for {nii_path}: {e}. Falling back.")\n            one_volume = torch.zeros(1, *target_size, dtype=torch.float32)\n            print("Warning: Failed to load volume. Using a zero tensor as a placeholder.")\n"""
replacement = """        except Exception as e:\n            if os.environ.get("CLINFUSION_STRICT_3D", "1").lower() not in ("0", "false", "no"):\n                raise RuntimeError(f"3D preprocessing failed for {nii_path}: {e}") from e\n            print(f"Preprocessing failed for {nii_path}: {e}. Falling back.")\n            one_volume = torch.zeros(1, *target_size, dtype=torch.float32)\n            print("Warning: Failed to load volume. Using a zero tensor as a placeholder.")\n"""
if needle in text:
    text = text.replace(needle, replacement, 1)
adapter.write_text(text)

text = openclip.read_text()
if "import deepspeed\n" in text:
    text = text.replace("import deepspeed\n", "try:\n    import deepspeed\nexcept ImportError:\n    deepspeed = None\n", 1)
openclip.write_text(text)

print("Patched ClinFusion for ROCm: SDPA/FP16 defaults, case-insensitive 32B selection, optional DeepSpeed, strict 3D errors.")
