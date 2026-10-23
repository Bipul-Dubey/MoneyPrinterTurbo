"""Metaso MiniMax H3 text-to-video client.

This module only handles the Metaso-proxied MiniMax V2 protocol: submitting paid tasks,
polling the same task, and parsing generation results. On-demand material generation, file downloading,
and final video stitching remain the responsibility of the ``material`` service, avoiding coupling
between provider protocols and local video workflows.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect


DEFAULT_BASE_URL = "https://metaso.cn/api/minimax"
DEFAULT_MODEL_ID = "MiniMax-H3"
DEFAULT_RESOLUTION = "2K"
DEFAULT_MIN_DURATION_SECONDS = 4
DEFAULT_MAX_DURATION_SECONDS = 15
DEFAULT_POLL_INTERVAL_SECONDS = 10.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
MAX_PROMPT_LENGTH = 7000
MAX_POLL_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
MAX_ERROR_TEXT_LENGTH = 500
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
ACTIVE_STATUSES = frozenset({"queued", "running"})
TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled", "canceled"})
SUPPORTED_RESOLUTIONS = frozenset({"768P", "2K"})


class MetasoMiniMaxError(RuntimeError):
    """Deterministic configuration, request, or response error from Metaso MiniMax."""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message)
        # Once a remote task is created, all subsequent exceptions carry the same task ID.
        # The task service can preserve recovery clues without needing to know which step failed.
        self.task_id = task_id


class MetasoMiniMaxUnconfirmedTaskError(MetasoMiniMaxError):
    """A paid task may have been created remotely, but its final status cannot be confirmed locally."""


class MetasoMiniMaxDownloadError(MetasoMiniMaxError):
    """The remote paid task succeeded, but the rendered video could not be downloaded locally."""


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """
    Read Metaso credentials following fixed priority order.

    Metaso ``mk-`` keys belong to a different account system than official MiniMax keys,
    so the project's existing ``minimax_api_key`` cannot be reused. Dedicated settings
    and environment variables also prevent unintentional credential changes when switching LLM providers.
    """
    settings = config.app if settings is None else settings
    configured = str(settings.get("metaso_minimax_api_key", "") or "").strip()
    environment_key = os.getenv("METASO_MINIMAX_API_KEY", "").strip()
    return configured or environment_key


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    """Return whether current configuration has valid credentials for Metaso video endpoints."""
    return bool(get_api_key(settings))


def _base_url() -> str:
    value = (
        str(
            config.app.get("metaso_minimax_base_url", DEFAULT_BASE_URL)
            or DEFAULT_BASE_URL
        )
        .strip()
        .rstrip("/")
    )
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MetasoMiniMaxError(
            "metaso_minimax_base_url must be an absolute HTTP(S) URL"
        )
    return value


def _resolution() -> str:
    configured = config.app.get("metaso_minimax_resolution", DEFAULT_RESOLUTION)
    value = str(configured).strip().upper()
    if value not in SUPPORTED_RESOLUTIONS:
        supported = ", ".join(sorted(SUPPORTED_RESOLUTIONS))
        # Resolution directly impacts generation costs; invalid user values must not silently fall back to 2K.
        # Defaults are only applied when the configuration is completely missing to avoid unexpected charges.
        raise MetasoMiniMaxError(
            f"Unsupported Metaso MiniMax resolution {value!r}; "
            f"expected one of: {supported}"
        )
    return value


def _tls_verify() -> bool:
    value = config.app.get("tls_verify", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _bounded_float(key: str, default: float, minimum: float, maximum: float) -> float:
    """Read a finite float setting bounded to a safe range that will not overload remote or local resources."""
    try:
        value = float(config.app.get(key, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, minimum), maximum)


def _status_code(response: Any) -> int:
    try:
        return int(getattr(response, "status_code", 200))
    except (TypeError, ValueError):
        return 200


def _redact_secret(value: Any, api_key: str) -> str:
    """Retain troubleshooting text while redacting API keys, URL-encoded keys, and proxy credentials."""
    text = str(value or "")
    if api_key:
        text = text.replace(api_key, "***")
        encoded = quote_plus(api_key)
        if encoded != api_key:
            text = text.replace(encoded, "***")
    for proxy_url in config.proxy.values():
        proxy_secret = str(proxy_url or "")
        if proxy_secret:
            text = text.replace(proxy_secret, "***")
    return text[:MAX_ERROR_TEXT_LENGTH]


def _response_error(response: Any, api_key: str) -> str:
    """Support MiniMax V2 nested error structures and limit response length in logs."""
    try:
        payload = response.json()
    except Exception:
        return f"HTTP {_status_code(response)}"
    if not isinstance(payload, dict):
        return f"HTTP {_status_code(response)}"

    error = payload.get("error")
    if isinstance(error, dict):
        error_type = error.get("type")
        message = error.get("message")
        http_code = error.get("http_code")
    else:
        error_type = None
        message = payload.get("message") or error
        http_code = payload.get("code")
    detail = ": ".join(
        str(item) for item in (error_type, http_code, message) if item not in (None, "")
    )
    return _redact_secret(detail or f"HTTP {_status_code(response)}", api_key)


def _is_retryable_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    response = getattr(error, "response", None)
    return response is not None and _status_code(response) in RETRYABLE_STATUS_CODES


def _normalize_duration(minimum_duration: int) -> tuple[int, int]:
    """Return (requested_duration, actual_submitted_duration) to explain the 4-second minimum constraint in logs."""
    try:
        requested = int(minimum_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetasoMiniMaxError(
            "Metaso MiniMax clip duration must be a positive integer"
        ) from exc
    if requested <= 0:
        raise MetasoMiniMaxError(
            "Metaso MiniMax clip duration must be a positive integer"
        )
    duration = min(
        max(requested, DEFAULT_MIN_DURATION_SECONDS),
        DEFAULT_MAX_DURATION_SECONDS,
    )
    return requested, duration


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    """Submit a Metaso MiniMax H3 text-to-video task and wait for downloadable results."""
    api_key = get_api_key()
    if not api_key:
        raise MetasoMiniMaxError("Metaso MiniMax requires an API key")

    term = str(search_term or "").strip()
    if not term:
        # An empty prompt usually indicates upstream script splitting failed. Paid endpoints
        # should not probe with invalid inputs, which could generate unusable billed materials.
        raise MetasoMiniMaxError("Metaso MiniMax search term must not be empty")
    if len(term) > MAX_PROMPT_LENGTH:
        raise MetasoMiniMaxError(
            f"Metaso MiniMax search term exceeds {MAX_PROMPT_LENGTH} characters"
        )

    aspect = VideoAspect(video_aspect)
    requested_duration, duration = _normalize_duration(minimum_duration)
    if duration != requested_duration:
        logger.info(
            "Metaso MiniMax clip duration adjusted to H3 limits: "
            f"requested={requested_duration}s, using={duration}s, "
            f"supported={DEFAULT_MIN_DURATION_SECONDS}-{DEFAULT_MAX_DURATION_SECONDS}s"
        )
    resolution = _resolution()
    payload = {
        "model": DEFAULT_MODEL_ID,
        "content": [{"type": "text", "text": term}],
        "resolution": resolution,
        "duration": duration,
        "ratio": aspect.value,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    base_url = _base_url()
    create_url = f"{base_url}/v2/video_generation"
    logger.info(
        "generating video with Metaso MiniMax H3: "
        f"resolution={resolution}, ratio={aspect.value}, duration={duration}s, "
        f"prompt_length={len(term)}"
    )

    # If POST times out or returns 5xx, the remote end may have already created and billed the task.
    # The API does not provide idempotency keys, so never retry automatically; upper layers abort to avoid double-billing.
    try:
        response = trace_http_call(
            "POST", requests.post,
            create_url,
            json=payload,
            headers=headers,
            proxies=config.proxy,
            verify=_tls_verify(),
            timeout=(30, 60),
            allow_redirects=False,
        )
    except Exception as exc:
        raise MetasoMiniMaxUnconfirmedTaskError(
            "Metaso MiniMax submission returned no response; a paid task may "
            "already exist remotely: "
            f"error={type(exc).__name__}, detail={_redact_secret(exc, api_key)}"
        ) from exc

    status_code = _status_code(response)
    if 300 <= status_code < 400:
        raise MetasoMiniMaxUnconfirmedTaskError(
            "Metaso MiniMax submission returned a redirect; the paid task state "
            "is unknown and the request was not replayed"
        )
    if status_code >= 500:
        raise MetasoMiniMaxUnconfirmedTaskError(
            f"Metaso MiniMax submission failed with HTTP {status_code}; a paid "
            "task may already exist remotely"
        )
    if not 200 <= status_code < 300:
        raise MetasoMiniMaxError(
            "Metaso MiniMax video generation request rejected: "
            f"HTTP {status_code}, {_response_error(response, api_key)}"
        )
    try:
        body = response.json()
    except Exception as exc:
        raise MetasoMiniMaxUnconfirmedTaskError(
            "Metaso MiniMax submission returned an unreadable response; a paid "
            f"task may already exist remotely: error={type(exc).__name__}"
        ) from exc

    task_id = str(body.get("task_id") or "").strip() if isinstance(body, dict) else ""
    if not task_id:
        raise MetasoMiniMaxUnconfirmedTaskError(
            "Metaso MiniMax accepted the submission without returning a task id"
        )
    logger.info(f"Metaso MiniMax paid task created: id={task_id}")

    task = _wait_for_task(
        task_id=task_id,
        base_url=base_url,
        headers=headers,
        api_key=api_key,
    )
    content = task.get("content")
    video_url = content.get("url") if isinstance(content, dict) else None
    if not isinstance(video_url, str) or not video_url.startswith(
        ("http://", "https://")
    ):
        raise MetasoMiniMaxError(
            f"Metaso MiniMax task succeeded without a downloadable video: id={task_id}",
            task_id=task_id,
        )

    actual_duration = task.get("duration", duration)
    try:
        actual_duration = int(actual_duration)
    except (TypeError, ValueError, OverflowError):
        actual_duration = duration
    if actual_duration <= 0:
        actual_duration = duration

    return [
        MaterialInfo(
            provider="metaso_minimax",
            url=video_url,
            duration=actual_duration,
            source_info={
                "provider": "metaso_minimax",
                "search_term": term,
                "asset_id": task_id,
                # MiniMax 2K/768P are profile names; the API makes no fixed pixel dimension guarantee.
                # Do not guess width/height; rely on downloaded file probing if exact dimensions are needed.
                "rendition": {"id": task_id},
            },
        )
    ]


def _wait_for_task(
    *,
    task_id: str,
    base_url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any]:
    """Poll the same paid task until it succeeds, explicitly fails, or status cannot be confirmed locally."""
    deadline = time.monotonic() + _bounded_float(
        "metaso_minimax_run_timeout",
        DEFAULT_RUN_TIMEOUT_SECONDS,
        60.0,
        7200.0,
    )
    poll_interval = _bounded_float(
        "metaso_minimax_poll_interval",
        DEFAULT_POLL_INTERVAL_SECONDS,
        1.0,
        60.0,
    )
    query_url = f"{base_url}/v2/query/video_generation/{quote(task_id, safe='')}"
    consecutive_failures = 0

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MetasoMiniMaxUnconfirmedTaskError(
                "Metaso MiniMax task is still running after the configured local "
                f"wait timeout: id={task_id}",
                task_id=task_id,
            )

        # Separate connect/read timeouts each use half the remaining time to ensure a single GET
        # does not deliberately overrun the overall task deadline. Polling ceases once expired.
        phase_timeout = max(min(remaining / 2.0, 30.0), 0.001)
        try:
            response = trace_http_call(
                "GET", requests.get,
                query_url,
                headers=headers,
                proxies=config.proxy,
                verify=_tls_verify(),
                timeout=(phase_timeout, phase_timeout),
                allow_redirects=False,
            )
            status_code = _status_code(response)
            if status_code in RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"HTTP {status_code}", response=response
                )
            if not 200 <= status_code < 300:
                raise MetasoMiniMaxUnconfirmedTaskError(
                    "Metaso MiniMax task status is unknown: "
                    f"http_status={status_code}, "
                    f"detail={_response_error(response, api_key)}",
                    task_id=task_id,
                )
            body = response.json()
            task = body.get("task") if isinstance(body, dict) else None
            if not isinstance(task, dict):
                raise MetasoMiniMaxUnconfirmedTaskError(
                    "Metaso MiniMax task status response is malformed",
                    task_id=task_id,
                )
        except MetasoMiniMaxUnconfirmedTaskError:
            raise
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise MetasoMiniMaxUnconfirmedTaskError(
                    "Metaso MiniMax polling failed and the paid task state is "
                    f"unknown: error={type(exc).__name__}, "
                    f"detail={_redact_secret(exc, api_key)}",
                    task_id=task_id,
                ) from exc

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MetasoMiniMaxUnconfirmedTaskError(
                    "Metaso MiniMax task is still running after the configured "
                    f"local wait timeout: id={task_id}",
                    task_id=task_id,
                ) from exc
            consecutive_failures += 1
            if consecutive_failures > MAX_POLL_RETRIES:
                raise MetasoMiniMaxUnconfirmedTaskError(
                    "Metaso MiniMax polling failed after retries; the paid task "
                    f"may still be running remotely: id={task_id}",
                    task_id=task_id,
                ) from exc
            delay = min(RETRY_BASE_SECONDS * consecutive_failures, remaining)
            logger.warning(
                "Metaso MiniMax polling hit a transient error; retrying the same "
                f"task: id={task_id}, attempt={consecutive_failures}/"
                f"{MAX_POLL_RETRIES}, retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        consecutive_failures = 0
        status = str(task.get("status") or "").strip().lower()
        logger.info(f"Metaso MiniMax task status: id={task_id}, status={status}")
        if status == "succeeded":
            return task
        if status in TERMINAL_FAILURE_STATUSES:
            raise MetasoMiniMaxError(
                "Metaso MiniMax task did not produce a video: "
                f"id={task_id}, status={status}, "
                f"detail={_redact_secret(task.get('error'), api_key)}",
                task_id=task_id,
            )
        if status not in ACTIVE_STATUSES:
            raise MetasoMiniMaxUnconfirmedTaskError(
                "Metaso MiniMax returned an unknown task status: "
                f"id={task_id}, status={status!r}",
                task_id=task_id,
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MetasoMiniMaxUnconfirmedTaskError(
                "Metaso MiniMax task is still running after the configured local "
                f"wait timeout: id={task_id}",
                task_id=task_id,
            )
        time.sleep(min(poll_interval, remaining))
