import asyncio
import base64
import binascii
import json
import os
import secrets
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from api.model_runtime import runtime

API_KEY = os.getenv("API_KEY", "")
DEFAULT_MAX = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "128"))
MAX_TOKENS = int(os.getenv("MAX_NEW_TOKENS_LIMIT", "4096"))
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "2048")) * 1024 * 1024
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "10"))
MODEL_VARIANT = os.getenv("MODEL_VARIANT", "8B").strip().upper()
MODEL_ID = f"clinfusion-{MODEL_VARIANT.lower()}"
DATA_ROOT = Path("/data").resolve()
UPLOAD_ROOT = Path("/tmp/clinfusion_uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    return value.strip()


async def auth(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    """Accept either X-API-Key or OpenAI-style Authorization: Bearer."""
    if not API_KEY:
        return
    candidates = [x_api_key, _bearer_token(authorization)]
    if not any(v and secrets.compare_digest(v, API_KEY) for v in candidates):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def allowed_data_path(p: str) -> str:
    path = Path(p).expanduser().resolve()
    if path != DATA_ROOT and DATA_ROOT not in path.parents:
        raise HTTPException(
            400,
            detail=f"Path must be inside {DATA_ROOT}; mount files with HOST_DATA_DIR or use /v1/infer/files",
        )
    if not path.is_file():
        raise HTTPException(400, detail=f"File not found: {path}")
    return str(path)


def validate_gen(max_new_tokens: int, temperature: float, top_p: float):
    if not 1 <= max_new_tokens <= MAX_TOKENS:
        raise HTTPException(400, detail=f"max_new_tokens must be 1..{MAX_TOKENS}")
    if temperature < 0:
        raise HTTPException(400, detail="temperature must be >= 0")
    if not 0 < top_p <= 1:
        raise HTTPException(400, detail="top_p must be > 0 and <= 1")


async def save_upload(upload: UploadFile, directory: Path) -> str:
    name = upload.filename or "upload.bin"
    lower = name.lower()
    suffix = ".nii.gz" if lower.endswith(".nii.gz") else (Path(name).suffix or ".bin")
    dest = directory / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(4 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413,
                    detail=f"Upload exceeds MAX_UPLOAD_MB ({MAX_UPLOAD // 1024 // 1024} MB)",
                )
            f.write(chunk)
    return str(dest)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("MODEL_AUTOLOAD", "1").lower() not in ("0", "false", "no"):
        if runtime.files_ready() and runtime.gpu_status().get("available"):
            try:
                await asyncio.to_thread(runtime.load)
            except Exception as e:
                print(f"Model autoload failed: {type(e).__name__}: {e}")
        else:
            print(
                "Model autoload skipped: model files incomplete or GPU unavailable. "
                "API remains up for status checks."
            )
    yield


app = FastAPI(
    title=f"ClinFusion-{MODEL_VARIANT} Strix Halo API",
    version="0.3.0",
    description=(
        f"ROCm/Strix-Halo deployment wrapper for Alibaba DAMO ClinFusion-{MODEL_VARIANT}. "
        "Includes OpenAI-compatible text/vision chat and an upload endpoint for NIfTI."
    ),
    lifespan=lifespan,
)

origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferRequest(BaseModel):
    prompt: str = Field(min_length=1)
    images: list[str] = []
    niftis: list[str] = []
    max_new_tokens: int = DEFAULT_MAX
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any = ""


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    stream_options: dict[str, Any] | None = None

    def output_limit(self) -> int:
        return int(self.max_completion_tokens or self.max_tokens or DEFAULT_MAX)


@app.get("/healthz")
def healthz():
    return {"ok": True, "model_loaded": runtime.adapter is not None, "version": "0.3.0", "model": MODEL_ID, "variant": MODEL_VARIANT}


@app.get("/v1/model/status", dependencies=[Depends(auth)])
def model_status():
    return runtime.status()


@app.post("/v1/model/load", dependencies=[Depends(auth)])
async def model_load():
    try:
        return await asyncio.to_thread(runtime.load)
    except Exception as e:
        raise HTTPException(503, detail=f"Model load failed: {type(e).__name__}: {e}")


@app.post("/v1/model/unload", dependencies=[Depends(auth)])
async def model_unload():
    return await asyncio.to_thread(runtime.unload)


@app.get("/v1/models", dependencies=[Depends(auth)])
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "Alibaba-DAMO-Academy",
            }
        ],
    }


@app.post("/v1/infer", dependencies=[Depends(auth)])
async def infer(req: InferRequest):
    validate_gen(req.max_new_tokens, req.temperature, req.top_p)
    images = [allowed_data_path(x) for x in req.images]
    niftis = [allowed_data_path(x) for x in req.niftis]
    try:
        result = await asyncio.to_thread(
            runtime.infer,
            req.prompt,
            images,
            niftis,
            req.max_new_tokens,
            req.temperature,
            req.top_p,
            req.repetition_penalty,
        )
        return {"model": MODEL_ID, **result}
    except Exception as e:
        raise HTTPException(500, detail=f"Inference failed: {type(e).__name__}: {e}")


@app.post("/v1/infer/files", dependencies=[Depends(auth)])
async def infer_files(
    prompt: str = Form(...),
    images: list[UploadFile] | None = File(default=None),
    niftis: list[UploadFile] | None = File(default=None),
    max_new_tokens: int = Form(DEFAULT_MAX),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.0),
):
    validate_gen(max_new_tokens, temperature, top_p)
    if len(images or []) > MAX_IMAGES:
        raise HTTPException(400, detail=f"At most {MAX_IMAGES} images are accepted per request")
    tmp = Path(tempfile.mkdtemp(prefix="req-", dir=UPLOAD_ROOT))
    try:
        image_paths = [await save_upload(x, tmp) for x in (images or [])]
        nifti_paths = [await save_upload(x, tmp) for x in (niftis or [])]
        result = await asyncio.to_thread(
            runtime.infer,
            prompt,
            image_paths,
            nifti_paths,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
        )
        return {"model": MODEL_ID, **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Inference failed: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _content_parts(content: Any) -> tuple[str, list[str]]:
    """Return text and image URLs/data URLs from OpenAI-style message content."""
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []
    texts: list[str] = []
    images: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type", ""))
        if kind in ("text", "input_text"):
            value = item.get("text", "")
            if value:
                texts.append(str(value))
        elif kind in ("image_url", "input_image"):
            value: Any = item.get("image_url", "")
            if isinstance(value, dict):
                value = value.get("url", "")
            if value:
                images.append(str(value))
    return " ".join(texts).strip(), images


def _flatten_chat_and_images(messages: list[ChatMessage]) -> tuple[str, list[str]]:
    parts: list[str] = []
    image_urls: list[str] = []
    for m in messages:
        text, urls = _content_parts(m.content)
        image_urls.extend(urls)
        if text:
            parts.append(f"{m.role}: {text}")
    prompt = "\n".join(parts).strip()
    if not prompt:
        raise HTTPException(400, detail="No text prompt found in messages")
    if len(image_urls) > MAX_IMAGES:
        raise HTTPException(400, detail=f"At most {MAX_IMAGES} images are accepted per request")
    return prompt, image_urls


def _decode_data_image(url: str, directory: Path) -> str:
    if not url.startswith("data:image/"):
        if url.startswith(("http://", "https://")):
            raise HTTPException(
                400,
                detail=(
                    "Remote image URLs are disabled for SSRF safety. Send images as data:image/...;base64 "
                    "(Open WebUI does this automatically for vision models)."
                ),
            )
        raise HTTPException(400, detail="Unsupported image URL; expected a data:image/...;base64 URL")
    header, sep, payload = url.partition(",")
    if not sep or ";base64" not in header.lower():
        raise HTTPException(400, detail="Image data URL must be base64 encoded")
    mime = header[5:].split(";", 1)[0].lower()
    ext_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/gif": ".gif",
    }
    ext = ext_map.get(mime, ".img")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(400, detail=f"Invalid base64 image: {e}") from e
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(
            413,
            detail=f"Image exceeds MAX_UPLOAD_MB ({MAX_UPLOAD // 1024 // 1024} MB)",
        )
    path = directory / f"openai-{uuid.uuid4().hex}{ext}"
    path.write_bytes(raw)
    return str(path)


def _chat_response(result: dict[str, Any]) -> dict[str, Any]:
    usage = result.get("usage", {})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["output"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": usage.get("completion_tokens") or 0,
            "total_tokens": usage.get("completion_tokens") or 0,
        },
        "metrics": result.get("metrics"),
    }


def _sse_payload(data: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def _completed_stream(result: dict[str, Any]):
    """OpenAI-compatible SSE framing. Generation itself is not token-streamed upstream."""
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    yield _sse_payload(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    yield _sse_payload(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": result["output"]},
                    "finish_reason": None,
                }
            ],
        }
    )
    yield _sse_payload(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": (result.get("usage") or {}).get("completion_tokens") or 0,
                "total_tokens": (result.get("usage") or {}).get("completion_tokens") or 0,
            },
        }
    )
    yield b"data: [DONE]\n\n"


@app.post("/v1/chat/completions", dependencies=[Depends(auth)])
async def chat_completions(req: ChatRequest):
    max_tokens = req.output_limit()
    validate_gen(max_tokens, req.temperature, req.top_p)
    prompt, image_urls = _flatten_chat_and_images(req.messages)
    tmp: Path | None = None
    try:
        image_paths: list[str] = []
        if image_urls:
            tmp = Path(tempfile.mkdtemp(prefix="openai-vision-", dir=UPLOAD_ROOT))
            image_paths = [_decode_data_image(url, tmp) for url in image_urls]
        result = await asyncio.to_thread(
            runtime.infer,
            prompt,
            image_paths,
            [],
            max_tokens,
            req.temperature,
            req.top_p,
            1.0,
        )
        if req.stream:
            return StreamingResponse(_completed_stream(result), media_type="text/event-stream")
        return _chat_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Inference failed: {type(e).__name__}: {e}")
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
