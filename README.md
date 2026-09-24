# ClinFusion 8B / 32B on AMD Strix Halo — Docker + API v3

Ready-to-run ROCm deployment wrapper for `alibaba-damo-academy/ClinFusion` on Ryzen AI Max / Max+ (gfx1151), aimed at Ubuntu 24.04 + ROCm 7.2.1.

**v3 keeps ClinFusion-8B as the default.** If you upgrade from the working v2 package and simply run `./start.sh`, it continues to use 8B on port **8008**. 32B is an opt-in future test.

## v3 highlights

- `MODEL_VARIANT=8B` by default
- Selectable **ClinFusion-8B** or **ClinFusion-32B** using the same API and Open WebUI integration
- Shared DINOv2 / ConvNeXt cache between 8B and 32B
- Check or pre-download 32B **without selecting it**
- Safe explicit switch command: `./select-model.sh 32B`
- Switching models never tries to keep both loaded at once
- API remains on `http://localhost:8008` by default
- OpenAI-compatible text + vision API and Open WebUI NIfTI Pipe retained from v2
- Fixes the upstream case-sensitive 32B base-model selector (`ClinFusion-32B` now correctly selects Qwen3-VL-32B)

## 1. Upgrade from your working v2 deployment

Extract v3 and copy your existing configuration:

```bash
unzip clinfusion-strix-halo.zip
cd clinfusion-strix-halo
./upgrade-from-v2.sh /path/to/clinfusion-strix-halo-v2
./start.sh
```

The upgrade helper adds only:

```bash
MODEL_VARIANT=8B
```

Your existing `HOST_CACHE_DIR`, API key, port, data path, and model files are reused.

v3 intentionally retains the v2 Compose project identity so `./start.sh` updates/recreates the same API service rather than trying to start a second container on port 8008.

If you copy v3 files over your existing directory instead, `./start.sh` automatically adds `MODEL_VARIANT=8B` when it is absent.

## 2. Normal/default operation — 8B

```bash
./start.sh
```

Check the active model:

```bash
curl -s http://localhost:8008/v1/model/status | jq '.variant, .model, .loaded'
```

Expected:

```text
"8B"
"clinfusion-8b"
true
```

The cache layout remains:

```text
cache/
  models/
    Qwen3-VL-8B-Instruct/
    ClinFusion-8B/
    dinov2-large/
    CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/
  3d_volume/
```

## 3. Prepare 32B without changing the active model

This is the recommended first step:

```bash
./models.sh 32B
```

It checks these additional folders:

```text
cache/models/Qwen3-VL-32B-Instruct/
cache/models/ClinFusion-32B/
```

DINOv2 and ConvNeXt are shared with 8B and are not duplicated.

If 32B files are missing/incomplete, the script asks whether to download/resume them. Existing Hugging Face partial downloads are reused.

Running `./models.sh 32B` **does not switch or restart the API**.

You can similarly check the current/default model with:

```bash
./models.sh
# or
./models.sh 8B
```

## 4. Switch to 32B later

When the files are ready:

```bash
./select-model.sh 32B
```

This only changes `.env`; it does not touch the currently running container. Then apply it:

```bash
./start.sh
```

Verify:

```bash
curl -s http://localhost:8008/v1/model/status | jq '.variant, .model, .gpu, .memory_advisory'
```

To return to the normal 8B deployment:

```bash
./select-model.sh 8B
./start.sh
```

## 5. Important 32B memory note for 128-GB Strix Halo

32B is optional and much closer to the memory ceiling. Before loading it, check the host's GPU-visible TTM/GTT allocation:

```bash
amd-ttm
```

On a 128-GB Strix Halo, roughly **100 GiB GPU-visible memory** is a sensible starting target for an unquantized FP16 32B test while leaving host RAM headroom. Do not run the 8B and 32B containers simultaneously.

The API status endpoint provides a warning if 32B is selected and PyTorch reports less than about 90 GiB GPU-visible memory.

Keep concurrency at one (the API already serializes GPU inference) and benchmark with short outputs first:

```bash
./scripts/benchmark.sh text
./scripts/benchmark.sh image /path/to/xray.png
./scripts/benchmark.sh nifti /path/to/scan.nii.gz
```

## 6. API endpoints

The selected model uses the same endpoints as v2:

```text
http://localhost:8008/healthz
http://localhost:8008/docs
http://localhost:8008/v1/models
http://localhost:8008/v1/model/status
http://localhost:8008/v1/model/load
http://localhost:8008/v1/model/unload
http://localhost:8008/v1/chat/completions
http://localhost:8008/v1/infer
http://localhost:8008/v1/infer/files
```

`GET /v1/models` reports whichever variant is currently configured:

```text
clinfusion-8b
```

or:

```text
clinfusion-32b
```

If `API_KEY` is set, use either `Authorization: Bearer YOUR_KEY` or `X-API-Key: YOUR_KEY`.

## 7. Open WebUI

The existing Open WebUI setup is retained. The included Pipe is now generically named:

```text
ClinFusion Medical 8B/32B (Strix Halo)
```

It follows whichever backend variant is currently selected, so you do not need a second Pipe for 32B.

Use:

```text
openwebui/clinfusion_pipe.py
```

It supports:

- text
- JPG / PNG / WebP and other normal medical images
- `.nii`
- `.nii.gz`
- follow-up prompts with attached medical files

Recommended Pipe URL when Open WebUI is Docker on the same host:

```text
http://host.docker.internal:8008
```

For a direct OpenAI-compatible connection, use:

```text
Base URL: http://<STRIX-IP>:8008/v1
```

After switching variants, Open WebUI's model discovery may need a refresh to show `clinfusion-32b` instead of `clinfusion-8b`. The Pipe itself does not need to change.

## 8. Text / image / NIfTI examples

Text:

```bash
curl -s http://localhost:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"clinfusion",
    "messages":[{"role":"user","content":"What is pneumonia?"}],
    "max_tokens":64
  }'
```

Image:

```bash
curl -s http://localhost:8008/v1/infer/files \
  -F 'prompt=Describe the important findings briefly.' \
  -F 'max_new_tokens=128' \
  -F 'images=@/path/to/xray.png'
```

NIfTI:

```bash
curl -s http://localhost:8008/v1/infer/files \
  -F 'prompt=Briefly describe the important findings in this CT volume.' \
  -F 'max_new_tokens=128' \
  -F 'niftis=@/path/to/scan.nii.gz'
```

## 9. Port customization

8008 remains the default. Change `.env` if needed:

```bash
API_PORT=8018
```

Update the Open WebUI connection/Pipe URL to match.

## 10. Useful commands

```bash
./start.sh                 # start selected model; defaults to 8B
./stop.sh
./logs.sh
./gpu-test.sh
./models.sh                # check selected/default model
./models.sh 8B
./models.sh 32B            # prepare/check 32B without switching
./select-model.sh 8B
./select-model.sh 32B
```

## 11. ROCm/upstream adaptations

The package retains all v2 ROCm changes:

- CUDA FlashAttention wheel is not installed
- PyTorch SDPA is used by default
- FP16 is the default dtype
- DeepSpeed import is optional for single-GPU inference
- Ray/vLLM are not required by the API wrapper
- strict 3D preprocessing errors are enabled by default instead of silently substituting a zero volume

v3 additionally fixes the upstream 32B selector. The upstream adapter uses a case-sensitive check for lowercase `32b`, while the official model path is `ClinFusion-32B`; v3 patches that selection to be case-insensitive before the container image is built.

## 12. Versions

- Package: v3.0
- API wrapper: 0.3.0
- Default model: ClinFusion-8B
- Optional model: ClinFusion-32B
- Default API port: 8008
- ClinFusion source ref: `96dd208ea7cdfecd001a703e232aa607802a8fd6`
- Transformers: `4.57.0`
- ROCm base: `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`
- Attention: PyTorch SDPA
- Default dtype: FP16

## Important

ClinFusion is a research medical model. This deployment wrapper does not make outputs clinically validated or suitable for autonomous patient-care decisions. Validate model behavior for your intended workflow.
