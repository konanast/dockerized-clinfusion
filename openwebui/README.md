# Open WebUI integration — ClinFusion v3

v3 uses the same Open WebUI integration as v2, but the Pipe is now model-size agnostic. It works with whichever backend is selected through `MODEL_VARIANT`.

## Recommended: Pipe (images + NIfTI)

Install/import:

```text
openwebui/clinfusion_pipe.py
```

The model appears as **ClinFusion Medical 8B/32B (Strix Halo)**.

Set its valves:

```text
API_BASE_URL = http://host.docker.internal:8008
API_KEY      = <same API_KEY as ClinFusion .env, or blank>
```

If Open WebUI is on another machine, use the Strix Halo LAN address instead.

The Pipe supports text, normal images, `.nii`, and `.nii.gz`. It sends NIfTI binaries to `/v1/infer/files`, so the file reaches ClinFusion's actual 3D pipeline rather than being treated as a RAG document.

The Pipe follows whichever model the backend currently runs. Switching from 8B to 32B does not require a new Pipe.

## Direct OpenAI connection (text + normal images)

Use:

```text
Base URL: http://<STRIX-IP>:8008/v1
API key:  <your API_KEY>
```

`GET /v1/models` returns `clinfusion-8b` or `clinfusion-32b` according to the active v3 selection. Refresh the Open WebUI connection/model list after switching variants if necessary.

For `.nii/.nii.gz`, use the Pipe because Chat Completions has no standard NIfTI content type.

## Test

```bash
./openwebui/test_connection.sh
```

or, with an API key:

```bash
./openwebui/test_connection.sh http://localhost:8008 YOUR_API_KEY
```
