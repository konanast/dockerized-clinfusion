#!/usr/bin/env python3
import time
import torch
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ROCm GPU not visible in container")
print("GPU:", torch.cuda.get_device_name(0))
print("Device capability:", torch.cuda.get_device_capability(0))
x = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    y = x @ x
torch.cuda.synchronize()
print(f"10 x FP16 4096x4096 GEMM: {time.perf_counter()-t0:.3f}s")
print("Result:", y.shape, y.device, y.dtype)
