"""
title: ClinFusion Medical 8B/32B (Strix Halo)
author: OpenAI deployment helper
version: 3.0.0
required_open_webui_version: 0.6.0
requirements: requests>=2.31
license: Apache-2.0
description: Open WebUI Pipe for ClinFusion text, medical images, and NIfTI .nii/.nii.gz volumes.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        API_BASE_URL: str = Field(
            default="http://host.docker.internal:8008",
            description="ClinFusion server URL without /v1, e.g. http://host.docker.internal:8008",
        )
        API_KEY: str = Field(default="", description="ClinFusion API_KEY from .env; blank if disabled")
        REQUEST_TIMEOUT_SECONDS: int = Field(default=1800, ge=30, le=7200)
        VERIFY_TLS: bool = Field(default=True)
        MAX_NEW_TOKENS: int = Field(default=256, ge=1, le=4096)
        TEMPERATURE: float = Field(default=0.0, ge=0.0)
        TOP_P: float = Field(default=1.0, gt=0.0, le=1.0)
        REJECT_UNSUPPORTED_FILES: bool = Field(
            default=True,
            description="Fail instead of pretending unsupported attached documents were seen by ClinFusion",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def _status(self, emitter, description: str, done: bool = False):
        if emitter:
            await emitter({"type": "status", "data": {"description": description, "done": done}})

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                value = part.get("text", "")
                if value:
                    out.append(str(value))
        return " ".join(out).strip()

    def _prompt(self, body: dict, metadata: dict | None) -> str:
        messages = body.get("messages") or []
        lines = []
        last_user_index = max(
            (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"),
            default=-1,
        )
        original_user_prompt = (metadata or {}).get("user_prompt") or ""
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user"))
            text = self._message_text(msg.get("content"))
            if i == last_user_index and original_user_prompt:
                text = original_user_prompt
            if text:
                lines.append(f"{role}: {text}")
        return "\n".join(lines).strip() or original_user_prompt or "Describe the provided medical input."

    @staticmethod
    def _image_data_urls(body: dict) -> list[str]:
        urls: list[str] = []
        for msg in body.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in ("image_url", "input_image"):
                    continue
                value = part.get("image_url", "")
                if isinstance(value, dict):
                    value = value.get("url", "")
                if isinstance(value, str) and value.startswith("data:image/"):
                    urls.append(value)
        return urls

    @staticmethod
    def _decode_data_url(data_url: str, index: int) -> tuple[str, bytes, str]:
        header, sep, payload = data_url.partition(",")
        if not sep or ";base64" not in header.lower():
            raise ValueError("Open WebUI image was not a base64 data URL")
        mime = header[5:].split(";", 1)[0] or "image/png"
        ext = mimetypes.guess_extension(mime) or ".png"
        return f"openwebui-image-{index}{ext}", base64.b64decode(payload), mime

    @staticmethod
    def _file_path(item: dict) -> Path:
        file_info = item.get("file") if isinstance(item, dict) else None
        candidates = []
        if isinstance(file_info, dict):
            candidates.extend([file_info.get("path"), file_info.get("filename")])
        if isinstance(item, dict):
            candidates.extend([item.get("path")])

        # Resolve Open WebUI storage paths if its storage provider is available.
        for candidate in candidates:
            if not candidate:
                continue
            try:
                from open_webui.storage.provider import Storage

                resolved = Storage.get_file(candidate)
                if resolved and Path(str(resolved)).is_file():
                    return Path(str(resolved))
            except Exception:
                pass
            p = Path(str(candidate))
            if p.is_file():
                return p
        name = (item or {}).get("name") or (file_info or {}).get("filename") or "unknown file"
        raise FileNotFoundError(f"Open WebUI attachment is not accessible on local storage: {name}")

    @staticmethod
    def _classify(path: Path) -> str | None:
        lower = path.name.lower()
        if lower.endswith((".nii", ".nii.gz")):
            return "nifti"
        if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
            return "image"
        return None

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.valves.API_KEY} if self.valves.API_KEY else {}

    def _post_files(self, prompt: str, body: dict, chat_files: list[dict]) -> str:
        multipart: list[tuple[str, tuple[str, Any, str]]] = []
        opened = []
        try:
            for i, data_url in enumerate(self._image_data_urls(body)):
                filename, raw, mime = self._decode_data_url(data_url, i)
                multipart.append(("images", (filename, raw, mime)))

            unsupported = []
            for item in chat_files:
                path = self._file_path(item)
                kind = self._classify(path)
                if not kind:
                    unsupported.append(path.name)
                    continue
                fh = path.open("rb")
                opened.append(fh)
                content_type = (
                    "application/gzip"
                    if path.name.lower().endswith(".nii.gz")
                    else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                )
                multipart.append(("niftis" if kind == "nifti" else "images", (path.name, fh, content_type)))

            if unsupported and self.valves.REJECT_UNSUPPORTED_FILES:
                raise ValueError(
                    "ClinFusion supports medical images and NIfTI volumes in this Pipe. "
                    f"Unsupported attachment(s): {', '.join(unsupported)}"
                )

            data = {
                "prompt": prompt,
                "max_new_tokens": str(self.valves.MAX_NEW_TOKENS),
                "temperature": str(self.valves.TEMPERATURE),
                "top_p": str(self.valves.TOP_P),
            }
            response = requests.post(
                f"{self.valves.API_BASE_URL.rstrip('/')}/v1/infer/files",
                headers=self._headers(),
                data=data,
                files=multipart,
                timeout=self.valves.REQUEST_TIMEOUT_SECONDS,
                verify=self.valves.VERIFY_TLS,
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("output") or str(payload)
        finally:
            for fh in opened:
                try:
                    fh.close()
                except Exception:
                    pass

    def _post_chat(self, body: dict) -> str:
        payload = {
            "model": "clinfusion",
            "messages": body.get("messages") or [],
            "max_tokens": self.valves.MAX_NEW_TOKENS,
            "temperature": self.valves.TEMPERATURE,
            "top_p": self.valves.TOP_P,
            "stream": False,
        }
        response = requests.post(
            f"{self.valves.API_BASE_URL.rstrip('/')}/v1/chat/completions",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=self.valves.REQUEST_TIMEOUT_SECONDS,
            verify=self.valves.VERIFY_TLS,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def pipe(
        self,
        body: dict,
        __files__: list[dict] | None = None,
        __metadata__: dict | None = None,
        __event_emitter__=None,
    ):
        chat_files = __files__ or []
        has_images = bool(self._image_data_urls(body))
        try:
            await self._status(__event_emitter__, "ClinFusion is analyzing the request…", False)
            if chat_files or has_images:
                prompt = self._prompt(body, __metadata__)
                result = await asyncio.to_thread(self._post_files, prompt, body, chat_files)
            else:
                result = await asyncio.to_thread(self._post_chat, body)
            await self._status(__event_emitter__, "ClinFusion analysis complete", True)
            return result
        except requests.RequestException as e:
            detail = ""
            response = getattr(e, "response", None)
            if response is not None:
                try:
                    detail = response.text[:2000]
                except Exception:
                    pass
            await self._status(__event_emitter__, "ClinFusion request failed", True)
            return f"ClinFusion API request failed: {e}{(': ' + detail) if detail else ''}"
        except Exception as e:
            await self._status(__event_emitter__, "ClinFusion request failed", True)
            return f"ClinFusion integration error: {type(e).__name__}: {e}"
