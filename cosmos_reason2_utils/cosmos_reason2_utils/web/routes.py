# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""API routes for the Cosmos-Reason2 web interface."""

import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import quote

import openai
from flask import Blueprint, current_app, jsonify, request, send_from_directory

from cosmos_reason2_utils.text import (
    REASONING_PROMPT,
    SYSTEM_PROMPT,
    create_conversation_openai,
)
from cosmos_reason2_utils.vision import PIXELS_PER_TOKEN

logger = logging.getLogger(__name__)

api = Blueprint("api", __name__)

HOSTFS_ROOT = "/hostfs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# Sampling defaults copied from SamplingOverrides.get_defaults() in script/inference.py.
# We avoid importing that module because it triggers init_script() and requires vLLM/GPU.
SAMPLING_DEFAULTS = {
    "reasoning": {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 4096,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
    },
    "non_reasoning": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 4096,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
    },
}

# Default FPS and pixel settings (from qwen_vl_utils defaults)
DEFAULT_FPS = 2.0
DEFAULT_MAX_MODEL_LEN = 16384


def _get_vllm_client() -> openai.OpenAI:
    """Create an OpenAI client pointing at the vLLM server."""
    vllm_url = current_app.config["VLLM_URL"]
    return openai.OpenAI(api_key="EMPTY", base_url=f"{vllm_url}/v1")


def _resolve_hostfs_path(path: str) -> str | None:
    """Resolve a path under HOSTFS_ROOT, returning None if it escapes."""
    resolved = os.path.realpath(os.path.join(HOSTFS_ROOT, path.lstrip("/")))
    if not resolved.startswith(HOSTFS_ROOT):
        return None
    return resolved


def _to_hostfs_path(path: str) -> str:
    """Ensure a file path has the /hostfs prefix for vLLM access."""
    if path.startswith(HOSTFS_ROOT):
        return path
    return os.path.join(HOSTFS_ROOT, path.lstrip("/"))


def _to_file_url(path: str) -> str:
    """Convert a path to a properly encoded file:// URL.

    Avoids issues with special characters like # ? % in filenames
    which would be misinterpreted as URL fragment/query delimiters.
    """
    hostfs_path = _to_hostfs_path(path)
    return "file://" + quote(hostfs_path, safe="/")


def _classify_entry(name: str) -> str:
    """Classify a filesystem entry by extension."""
    ext = os.path.splitext(name)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "file"


# ── Workspace persistence ────────────────────────────────────────────────────


def _workspaces_path() -> str:
    cache_dir = current_app.config.get("WORKSPACE_CACHE", "/workspace_cache")
    return os.path.join(cache_dir, "workspaces.json")


def _load_workspaces() -> list[dict]:
    path = _workspaces_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_workspaces(workspaces: list[dict]) -> None:
    path = _workspaces_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(workspaces, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _next_workspace_name(workspaces: list[dict]) -> str:
    r"""Generate the next default workspace name.

    Given existing workspaces ["workspace0", "IR1", "workspace1"],
    returns "workspace2" (max N from ``workspace\d+`` pattern + 1).
    """
    max_n = -1
    for ws in workspaces:
        m = re.match(r"^workspace(\d+)$", ws.get("name", ""))
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"workspace{max_n + 1}"


# ── Page routes ──────────────────────────────────────────────────────────────


@api.route("/interview")
def interview():
    """Serve the SPA shell."""
    return send_from_directory(current_app.static_folder, "index.html")


# ── API routes ───────────────────────────────────────────────────────────────


# ── Workspace CRUD ───────────────────────────────────────────────────────────


@api.route("/api/workspaces")
def list_workspaces():
    """List all saved workspaces."""
    return jsonify({"workspaces": _load_workspaces()})


@api.route("/api/workspaces", methods=["POST"])
def create_workspace():
    """Create a new workspace with an auto-generated or user-supplied name."""
    data = request.get_json(force=True) if request.is_json else {}
    workspaces = _load_workspaces()
    raw_name = data.get("name", "")
    if raw_name and not isinstance(raw_name, str):
        return jsonify({"error": "name must be a string"}), 400
    name = (raw_name[:256].strip() if isinstance(raw_name, str) else "") or _next_workspace_name(workspaces)
    folders = data.get("folders", [])
    if not isinstance(folders, list) or not all(isinstance(f, str) for f in folders):
        return jsonify({"error": "folders must be a list of strings"}), 400
    if len(folders) > 100:
        return jsonify({"error": "Too many folders (max 100)"}), 400
    if any(len(f) > 4096 for f in folders):
        return jsonify({"error": "Folder path too long (max 4096)"}), 400
    if len(workspaces) >= 500:
        return jsonify({"error": "Too many workspaces (max 500)"}), 400
    now = datetime.now(timezone.utc).isoformat()
    ws = {
        "id": str(uuid.uuid4()),
        "name": name,
        "folders": folders,
        "created_at": now,
        "updated_at": now,
    }
    workspaces.append(ws)
    _save_workspaces(workspaces)
    return jsonify(ws), 201


@api.route("/api/workspaces/<ws_id>")
def get_workspace(ws_id):
    """Get a single workspace by ID."""
    for ws in _load_workspaces():
        if ws["id"] == ws_id:
            return jsonify(ws)
    return jsonify({"error": "Workspace not found"}), 404


@api.route("/api/workspaces/<ws_id>", methods=["PUT"])
def update_workspace(ws_id):
    """Update workspace name and/or folders."""
    data = request.get_json(force=True)
    if "name" in data:
        if not isinstance(data["name"], str):
            return jsonify({"error": "name must be a string"}), 400
        data["name"] = data["name"][:256].strip()
    if "folders" in data:
        if not isinstance(data["folders"], list) or not all(
            isinstance(f, str) for f in data["folders"]
        ):
            return jsonify({"error": "folders must be a list of strings"}), 400
        if len(data["folders"]) > 100:
            return jsonify({"error": "Too many folders (max 100)"}), 400
        if any(len(f) > 4096 for f in data["folders"]):
            return jsonify({"error": "Folder path too long (max 4096)"}), 400
    workspaces = _load_workspaces()
    for ws in workspaces:
        if ws["id"] == ws_id:
            if "name" in data:
                ws["name"] = data["name"]
            if "folders" in data:
                ws["folders"] = data["folders"]
            ws["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save_workspaces(workspaces)
            return jsonify(ws)
    return jsonify({"error": "Workspace not found"}), 404


@api.route("/api/workspaces/<ws_id>", methods=["DELETE"])
def delete_workspace(ws_id):
    """Delete a workspace."""
    workspaces = _load_workspaces()
    before = len(workspaces)
    workspaces = [ws for ws in workspaces if ws["id"] != ws_id]
    if len(workspaces) == before:
        return jsonify({"error": "Workspace not found"}), 404
    _save_workspaces(workspaces)
    return jsonify({"ok": True})


@api.route("/api/health")
def health():
    """Health check — reports vLLM availability and loaded model."""
    try:
        client = _get_vllm_client()
        models = client.models.list()
        model_id = models.data[0].id if models.data else "unknown"
        return jsonify({"vllm": "ok", "model": model_id})
    except Exception as exc:
        logger.warning("vLLM health check failed: %s", exc)
        return jsonify({"vllm": "unavailable", "error": str(exc)})


@api.route("/api/models")
def models():
    """Proxy vLLM /v1/models."""
    try:
        client = _get_vllm_client()
        result = client.models.list()
        return jsonify({"models": [m.id for m in result.data]})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@api.route("/api/defaults")
def defaults():
    """Return sampling defaults for reasoning/non-reasoning modes."""
    reasoning = request.args.get("reasoning", "false").lower() == "true"
    key = "reasoning" if reasoning else "non_reasoning"
    return jsonify(SAMPLING_DEFAULTS[key] | {"fps": DEFAULT_FPS})


@api.route("/api/browse")
def browse():
    """Browse the host filesystem under /hostfs.

    Query params:
        path: directory path (relative to host root)
    """
    raw_path = request.args.get("path", "/")
    resolved = _resolve_hostfs_path(raw_path)
    if resolved is None:
        return jsonify({"error": "Path traversal not allowed"}), 403
    if not os.path.isdir(resolved):
        return jsonify({"error": f"Not a directory: {raw_path}"}), 404

    entries = []
    try:
        for name in sorted(os.listdir(resolved)):
            full = os.path.join(resolved, name)
            if os.path.isdir(full):
                entries.append({"name": name, "type": "dir"})
            else:
                entry_type = _classify_entry(name)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = 0
                entries.append({"name": name, "type": entry_type, "size": size})
    except PermissionError:
        return jsonify({"error": f"Permission denied: {raw_path}"}), 403

    # Return the display path (relative to host root, not /hostfs)
    display_path = resolved.removeprefix(HOSTFS_ROOT) or "/"
    return jsonify({"path": display_path, "entries": entries})


@api.route("/api/media")
def media():
    """Serve a media file from the host filesystem for browser preview."""
    raw_path = request.args.get("path", "")
    resolved = _resolve_hostfs_path(raw_path)
    if resolved is None:
        return jsonify({"error": "Path traversal not allowed"}), 403
    if not os.path.isfile(resolved):
        return jsonify({"error": "Not a file"}), 404
    directory = os.path.dirname(resolved)
    filename = os.path.basename(resolved)
    return send_from_directory(directory, filename)


@api.route("/api/estimate-tokens", methods=["POST"])
def estimate_tokens():
    """Estimate token count for the given inputs.

    This is an approximation based on image/video resolution and PIXELS_PER_TOKEN.
    Actual tokenization happens inside vLLM with the full Qwen3-VL processor.
    """
    data = request.get_json(force=True)
    images = data.get("images", [])
    videos = data.get("videos", [])
    prompt = data.get("prompt", "")
    fps = data.get("fps", DEFAULT_FPS)
    max_model_len = data.get("max_model_len", DEFAULT_MAX_MODEL_LEN)
    max_tokens = data.get("max_tokens", SAMPLING_DEFAULTS["reasoning"]["max_tokens"])

    image_tokens = 0
    video_tokens = 0
    text_tokens = max(1, len(prompt) // 4) if prompt else 0

    # Estimate image tokens
    for img_path in images:
        hostfs_path = _to_hostfs_path(img_path)
        try:
            from PIL import Image

            with Image.open(hostfs_path) as im:
                w, h = im.size
            pixels = w * h
            image_tokens += max(1, pixels // PIXELS_PER_TOKEN)
        except Exception:
            image_tokens += 256  # fallback estimate

    # Estimate video tokens
    for vid_path in videos:
        hostfs_path = _to_hostfs_path(vid_path)
        try:
            import av

            with av.open(hostfs_path) as container:
                stream = container.streams.video[0]
                duration = float(stream.duration * stream.time_base)
                frame_w = stream.width
                frame_h = stream.height
            n_frames = max(1, int(duration * fps))
            pixels_per_frame = frame_w * frame_h
            video_tokens += n_frames * max(1, pixels_per_frame // PIXELS_PER_TOKEN)
        except Exception:
            video_tokens += 1024  # fallback estimate

    total_tokens = image_tokens + video_tokens + text_tokens
    budget_remaining = max_model_len - max_tokens - total_tokens

    return jsonify({
        "image_tokens": image_tokens,
        "video_tokens": video_tokens,
        "text_tokens": text_tokens,
        "total_tokens": total_tokens,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "budget_remaining": budget_remaining,
    })


@api.route("/api/infer", methods=["POST"])
def infer():
    """Run inference via the vLLM OpenAI-compatible API.

    Request JSON:
        prompt: str (required)
        images: list[str] — file paths on the host
        videos: list[str] — file paths on the host
        system_prompt: str
        reasoning: bool
        temperature: float
        max_tokens: int
        top_p: float
        top_k: int
        fps: float
        max_model_len: int
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # Files
    images = [_to_file_url(p) for p in data.get("images", [])]
    videos = [_to_file_url(p) for p in data.get("videos", [])]

    # Prompts
    reasoning = data.get("reasoning", False)
    system_prompt = data.get("system_prompt", SYSTEM_PROMPT)
    user_prompt = prompt
    if reasoning:
        user_prompt += f"\n\n{REASONING_PROMPT}"

    # Sampling parameters
    mode_defaults = SAMPLING_DEFAULTS["reasoning" if reasoning else "non_reasoning"]
    temperature = data.get("temperature", mode_defaults["temperature"])
    max_tokens = data.get("max_tokens", mode_defaults["max_tokens"])
    top_p = data.get("top_p", mode_defaults["top_p"])
    top_k = data.get("top_k", mode_defaults["top_k"])
    repetition_penalty = data.get(
        "repetition_penalty", mode_defaults["repetition_penalty"]
    )
    presence_penalty = data.get("presence_penalty", mode_defaults["presence_penalty"])

    # Vision parameters
    fps = data.get("fps", DEFAULT_FPS)
    max_model_len = data.get("max_model_len", DEFAULT_MAX_MODEL_LEN)

    # Build conversation (reuse existing utility)
    conversation = create_conversation_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        videos=videos,
    )

    sampling_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "presence_penalty": presence_penalty,
    }

    try:
        client = _get_vllm_client()

        # Resolve model
        model_name = data.get("model")
        if not model_name:
            models_list = client.models.list()
            if not models_list.data:
                return jsonify({"error": "No models loaded in vLLM"}), 502
            model_name = models_list.data[0].id

        start = time.time()
        chat_completion = client.chat.completions.create(
            messages=conversation,
            model=model_name,
            extra_body=sampling_kwargs,
        )
        duration_s = round(time.time() - start, 2)

        choice = chat_completion.choices[0]
        usage = chat_completion.usage

        return jsonify({
            "content": choice.message.content or "",
            "reasoning_content": getattr(choice.message, "reasoning_content", "") or "",
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "duration_s": duration_s,
        })

    except openai.APIConnectionError as exc:
        logger.error("vLLM connection error: %s", exc)
        return jsonify({"error": "Cannot connect to vLLM server"}), 502
    except openai.APIError as exc:
        logger.error("vLLM API error: %s", exc)
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        logger.exception("Inference error")
        return jsonify({"error": str(exc)}), 500
