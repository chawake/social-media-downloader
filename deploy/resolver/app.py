"""
Media Resolver Microservice for JapiAgent.
Self-hosted service with FastAPI + yt-dlp + FFmpeg deployed via Docker on Coolify.

Endpoints:
  POST /extract-video  Extracts video from TikTok and Instagram Reels.
  GET  /t?d=<tok>      Streams the resolved media with HMAC-SHA256 signed token + 15m expiration.
  GET  /health         Health check and liveness probe.
"""

import base64
from datetime import datetime, timezone
import glob
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Any, Optional
from urllib.parse import quote, urlparse

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.background import BackgroundTask
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from yt_dlp import YoutubeDL

# Disable Swagger UI, ReDoc, and OpenAPI endpoints in production
app = FastAPI(
    title="Japi Media Resolver",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

MAX_DURATION_SECONDS = 90
MAX_BYTES = 104_857_600  # 100 MB
TOKEN_TTL_SECONDS = 900  # 15 minutes
ALLOWED_SOURCE_HOSTS = {
    "tiktok.com",
    "instagram.com",
    "youtube.com",
    "youtu.be",
}

BASE_URL = (
    os.environ.get("BASE_URL")
    or os.environ.get("RENDER_EXTERNAL_URL")
    or "http://localhost:8080"
).rstrip("/")


def _public_base(request: Request) -> str:
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if forwarded_host:
        host = forwarded_host.split(",")[0].strip()
        proto = "http" if host.split(":")[0] in ("localhost", "127.0.0.1") else "https"
        return f"{proto}://{host}".rstrip("/")
    return BASE_URL


API_KEY = os.environ.get("RESOLVER_API_KEY", "").strip()
RESOLVER_SECRET = os.environ.get("RESOLVER_SECRET", "").strip()

# Enforce explicit credentials in production (fail-fast on startup)
IS_LOCAL = os.environ.get("ENV", "").lower() in ("local", "dev", "development") or BASE_URL.startswith("http://localhost")

if not IS_LOCAL:
    if not API_KEY:
        raise RuntimeError("FATAL: RESOLVER_API_KEY environment variable is required in production.")
    if not RESOLVER_SECRET:
        raise RuntimeError("FATAL: RESOLVER_SECRET environment variable is required in production.")

SECRET = (RESOLVER_SECRET or secrets.token_hex(32)).encode()

# Optional cookies for restricted sources
COOKIEFILE: Optional[str] = None
_cookies = os.environ.get("RESOLVER_COOKIES", "")
if _cookies.strip():
    COOKIEFILE = "/tmp/cookies.txt"
    with open(COOKIEFILE, "w", encoding="utf-8") as fh:
        fh.write(_cookies)

PROXY = os.environ.get("RESOLVER_PROXY", "").strip()

try:
    from yt_dlp.networking.impersonate import ImpersonateTarget

    _IMPERSONATE = ImpersonateTarget.from_str("chrome")
except Exception:
    _IMPERSONATE = None


def _base_opts() -> dict:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "max_filesize": MAX_BYTES,
    }
    if COOKIEFILE:
        opts["cookiefile"] = COOKIEFILE
    if PROXY:
        opts["proxy"] = PROXY
    if _IMPERSONATE is not None:
        opts["impersonate"] = _IMPERSONATE
    return opts


def _format_for(audio: bool, quality: Optional[str]) -> str:
    if audio:
        return "bestaudio/best"
    if quality == "480":
        return (
            "best[height<=480][ext=mp4][protocol^=http]/"
            "best[height<=480][protocol^=http]/best[ext=mp4]/best"
        )
    return "best[ext=mp4][protocol^=http]/best[protocol^=http]/best[ext=mp4]/best"


def _sign(payload: dict) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    # Full 64-char HMAC-SHA256 signature
    sig = hmac.new(SECRET, raw.encode(), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def _unsign(token: str) -> Optional[dict]:
    try:
        raw, sig = token.rsplit(".", 1)
    except ValueError:
        return None

    expect = hmac.new(SECRET, raw.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expect):
        return None

    pad = "=" * (-len(raw) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(raw + pad))
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    # Strict expiration check: tokens without exp or expired tokens are rejected
    exp = data.get("exp")
    if not exp or not isinstance(exp, (int, float)) or time.time() > exp:
        return None

    return data


def _safe_name(title: str, ext: str) -> str:
    stem = re.sub(r'[\\/:*?"<>|]+', "_", (title or "media")).strip() or "media"
    return f"{stem[:80]}.{ext}"


def _disposition(filename: str) -> str:
    ascii_name = filename.encode("ascii", "ignore").decode("ascii").strip() or "media"
    encoded = quote(filename, safe="")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{encoded}"


_LAST_ERR: str = ""
_LAST_ERROR_CODE: str = ""
_LAST_ERROR_DURATION: int = 0


def _validate_source_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")

    if parsed.scheme != "https" or not host:
        return "Only HTTPS source URLs are supported"

    if not any(host == allowed or host.endswith(f".{allowed}") for allowed in ALLOWED_SOURCE_HOSTS):
        return "Only public TikTok, Instagram, and YouTube URLs are supported"

    try:
        addresses = {record[4][0] for record in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
    except socket.gaierror:
        return "Could not resolve the source host"

    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        return "The source host resolves to a non-public address"

    return None


def _extract(url: str, audio: bool, quality: Optional[str]) -> Optional[dict]:
    global _LAST_ERR, _LAST_ERROR_CODE, _LAST_ERROR_DURATION
    _LAST_ERR = ""
    _LAST_ERROR_CODE = ""
    _LAST_ERROR_DURATION = 0
    opts = _base_opts()
    opts["format"] = _format_for(audio, quality)
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        _LAST_ERR = f"{type(exc).__name__}: {exc}"
        _LAST_ERROR_CODE = "resolution_failed"
        return None
    if not info:
        return None

    if info.get("_type") == "playlist" and info.get("entries"):
        entries = [e for e in info["entries"] if e]
        if not entries:
            return None
        info = entries[0]

    # Pre-check 1: Duration limit (max 90s)
    duration = int(info.get("duration") or 0)
    if duration > MAX_DURATION_SECONDS:
        _LAST_ERR = f"Duration {duration}s exceeds {MAX_DURATION_SECONDS}s limit for reference videos"
        _LAST_ERROR_CODE = "duration_exceeded"
        _LAST_ERROR_DURATION = duration
        return None

    # Pre-check 2: Filesize limit (max 100 MB) if known ahead of time
    filesize = info.get("filesize") or info.get("filesize_approx")
    if filesize and filesize > MAX_BYTES:
        _LAST_ERR = f"File size {filesize} bytes exceeds {MAX_BYTES} bytes (100 MB) limit"
        _LAST_ERROR_CODE = "size_exceeded"
        return None

    protocol = str(info.get("protocol") or "")
    direct = info.get("url")
    progressive = bool(direct) and protocol.startswith("http") and "m3u8" not in protocol
    return {
        "title": info.get("title") or "media",
        "duration": duration,
        "direct": direct,
        "headers": info.get("http_headers") or {},
        "ext": "mp3" if audio else (info.get("ext") or "mp4"),
        "progressive": progressive and not audio,
    }


def _authorized(request: Request) -> bool:
    if not API_KEY:
        return IS_LOCAL
    auth = request.headers.get("authorization", "").strip()
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
    elif auth.startswith("Api-Key "):
        token = auth[8:].strip()
    else:
        token = auth
    return hmac.compare_digest(token, API_KEY)


@app.get("/health")
def health(request: Request) -> JSONResponse:
    out: dict[str, Any] = {
        "status": "ok",
        "service": "japi-media-resolver",
        "auth": bool(COOKIEFILE),
        "proxy": bool(PROXY),
        "max_duration": MAX_DURATION_SECONDS,
        "max_bytes": MAX_BYTES,
    }
    return JSONResponse(out)


@app.post("/extract-video")
async def extract_video(request: Request) -> JSONResponse:
    if not _authorized(request):
        return JSONResponse({"status": "error", "error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        body = {}

    url = (body or {}).get("url")
    if not url or not isinstance(url, str):
        return JSONResponse({"status": "error", "error": "Invalid URL provided"}, status_code=400)

    url_error = _validate_source_url(url)
    if url_error:
        return JSONResponse({"status": "error", "error": {"code": "invalid_source", "message": url_error}}, status_code=400)

    audio = (body or {}).get("downloadMode") == "audio"
    quality = (body or {}).get("videoQuality")

    meta = _extract(url, audio, quality)
    if not meta or (not meta["direct"] and not meta["progressive"] and not audio):
        error_msg = _LAST_ERR or "Could not resolve video stream"
        error = {"code": _LAST_ERROR_CODE or "resolution_failed", "message": error_msg}
        if _LAST_ERROR_DURATION:
            error["duration_seconds"] = _LAST_ERROR_DURATION
        return JSONResponse({"status": "error", "error": error}, status_code=422)

    exp_timestamp = int(time.time()) + TOKEN_TTL_SECONDS
    expires_at = datetime.fromtimestamp(exp_timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    token = _sign({"u": url, "a": 1 if audio else 0, "q": quality or "", "exp": exp_timestamp})
    tunnel_url = f"{_public_base(request)}/t?d={token}"

    # Return clean, unified contract expected by Laravel
    return JSONResponse(
        {
            "media_url": tunnel_url,
            "mime_type": "audio/mpeg" if audio else "video/mp4",
            "duration_seconds": meta.get("duration", 0),
            "expires_at": expires_at,
            "title": meta.get("title", "media"),
        }
    )


_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "no-store",
}


def _pipe(url: str, audio: bool, quality: Optional[str]):
    fmt = _format_for(audio, quality)
    cmd = ["yt-dlp", "-f", fmt, "-o", "-", "--no-playlist", "--quiet"]
    if audio:
        cmd += ["-x", "--audio-format", "mp3"]
    else:
        cmd += ["--remux-video", "mp4"]
    if COOKIEFILE:
        cmd += ["--cookies", COOKIEFILE]
    if PROXY:
        cmd += ["--proxy", PROXY]
    if _IMPERSONATE is not None:
        cmd += ["--impersonate", "chrome"]
    cmd.append(url)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    bytes_sent = 0
    try:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            bytes_sent += len(chunk)
            if bytes_sent > MAX_BYTES:
                # Abort immediately: do not deliver a corrupt oversized partial file
                proc.kill()
                raise HTTPException(status_code=422, detail="Stream exceeded 100 MB limit")
            yield chunk
    finally:
        proc.stdout.close()
        proc.kill()


def _download_limited_file(url: str, quality: Optional[str]) -> tuple[str, str]:
    temp_dir = tempfile.mkdtemp(prefix="japi_reference_")
    output_template = os.path.join(temp_dir, "reference.%(ext)s")
    downloaded = 0

    def progress(status: dict) -> None:
        nonlocal downloaded
        downloaded = int(status.get("downloaded_bytes") or downloaded)
        if downloaded > MAX_BYTES:
            raise RuntimeError("Video exceeds 100 MB limit")

    options = _base_opts()
    options.update({
        "skip_download": False,
        "format": _format_for(False, quality),
        "outtmpl": output_template,
        "progress_hooks": [progress],
        "merge_output_format": "mp4",
    })

    try:
        with YoutubeDL(options) as ydl:
            ydl.extract_info(url, download=True)

        files = [path for path in glob.glob(os.path.join(temp_dir, "reference.*")) if os.path.isfile(path)]
        if len(files) != 1 or os.path.getsize(files[0]) == 0:
            raise RuntimeError("Could not create a complete reference video file")
        if os.path.getsize(files[0]) > MAX_BYTES:
            raise RuntimeError("Video exceeds 100 MB limit")

        return files[0], temp_dir
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


@app.get("/t")
def tunnel(d: str, request: Request) -> Any:
    data = _unsign(d)
    if not data:
        return JSONResponse({"error": "Bad or expired token"}, status_code=403)

    url = data["u"]
    audio = bool(data.get("a"))
    quality = data.get("q") or None
    meta = _extract(url, audio, quality)
    if not meta:
        return JSONResponse({"error": "Resolve failed or video limits exceeded"}, status_code=502)

    if audio:
        return JSONResponse({"error": "Audio tunnel downloads are not available"}, status_code=422)

    try:
        file_path, temp_dir = _download_limited_file(url, quality)
    except Exception as error:
        return JSONResponse({"error": str(error)}, status_code=422)

    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=_safe_name(meta["title"], "mp4"),
        background=BackgroundTask(shutil.rmtree, temp_dir, True),
    )
