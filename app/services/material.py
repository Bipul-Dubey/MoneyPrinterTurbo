import base64
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
import random
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, List
from urllib.parse import quote_plus, urlencode, urlsplit, urlunsplit

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image, UnidentifiedImageError

from app.config import config
from app.models.schema import ImageMotion, MaterialInfo, VideoAspect, VideoConcatMode
from app.services import (
    material_cache,
    material_upload,
    metaso_minimax,
    muapi,
    ofox,
    task_artifacts,
    video,
    volcengine_seedance,
)
from app.utils import logging_utils, utils

# Thread-safe counter for API key rotation
_api_key_counter = 0
_api_key_lock = threading.Lock()

# A provider URL can point to an unexpectedly large object or never-ending stream.
# Short stock and generated clips should stay well below this conservative cap.
MAX_VIDEO_DOWNLOAD_BYTES = 512 * 1024 * 1024

# Default to sequential execution to maintain legacy behavior; stock material concurrency can be increased in configuration if needed.
_DEFAULT_MATERIAL_CONCURRENCY = 1


def _get_material_concurrency() -> int:
    try:
        concurrency = int(config.app.get("material_concurrency", _DEFAULT_MATERIAL_CONCURRENCY))
    except (TypeError, ValueError):
        concurrency = _DEFAULT_MATERIAL_CONCURRENCY
    return max(1, min(8, concurrency))


class _OpenAIImageDecodeError(ValueError):
    """Indicates that bytes returned from the compatible API cannot be decoded as an image, excluding local file write errors."""


class OpenAIImagePaidResultError(RuntimeError):
    """A paid image request cannot provide a usable local video material."""


class OpenAIImageUnconfirmedError(OpenAIImagePaidResultError):
    """A paid image request may have succeeded without returning a response."""


def _safe_public_url(value: Any) -> str | None:
    """
    Retain only publicly displayable HTTP(S) page URLs, stripping query parameters and credentials.

    Material download URLs may carry API keys, signed JWTs, or temporary tokens. The task manifest only needs
    to help users navigate back to the provider's public material page and should not persist authentication parameters;
    URLs containing user credentials are also rejected to prevent persisting content like ``https://user:pass@example.com``.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _creator_info(value: Any) -> dict[str, str] | None:
    """Extract unified public fields from author structures across different providers."""
    if isinstance(value, str) and value.strip():
        return {"name": value.strip()}
    if not isinstance(value, dict):
        return None

    creator: dict[str, str] = {}
    creator_id = value.get("id")
    creator_name = value.get("name") or value.get("username")
    creator_page = _safe_public_url(
        value.get("url") or value.get("profile_url") or value.get("profile_page")
    )
    if creator_id is not None:
        creator["id"] = str(creator_id)
    if creator_name:
        creator["name"] = str(creator_name)
    if creator_page:
        creator["profile_page"] = creator_page
    return creator or None


def _material_source_record(item: MaterialInfo, local_path: str) -> dict[str, Any]:
    """
    Generate a lightweight provenance record for successfully downloaded materials.

    ``source_info`` may come from cache or an externally constructed ``MaterialInfo``, so
    it cannot be written as-is. It is reconstructed here using an allowlist, preserving only public pages,
    business identifiers, and dimensions, while recording only the local filename to avoid persisting user
    directories or Docker mount paths into task files.
    """
    source = item.source_info if isinstance(item.source_info, dict) else {}
    record: dict[str, Any] = {
        "provider": str(item.provider or source.get("provider") or ""),
        "local_file": Path(local_path).name,
        "duration": int(item.duration),
    }

    search_term = source.get("search_term")
    asset_id = source.get("asset_id")
    source_page = _safe_public_url(source.get("source_page"))
    if isinstance(search_term, str) and search_term.strip():
        record["search_term"] = search_term.strip()
    if asset_id not in (None, ""):
        record["asset_id"] = str(asset_id)
    if source_page:
        record["source_page"] = source_page

    creator = _creator_info(source.get("creator"))
    if creator:
        record["creator"] = creator

    raw_rendition = source.get("rendition")
    if isinstance(raw_rendition, dict):
        rendition = {}
        for field in ("id", "width", "height"):
            value = raw_rendition.get(field)
            if value not in (None, ""):
                rendition[field] = str(value) if field == "id" else value
        if rendition:
            record["rendition"] = rendition
    return record


def _persist_material_sources(
    task_id: str,
    material_sources: list[dict[str, Any]],
) -> None:
    """
    Append provenance for currently successfully downloaded materials to the task manifest.

    Task recording is an auxiliary capability; it must not alter the return value of video download functions,
    nor interrupt the main video generation flow due to disk write failures. ``patch_script_data`` handles atomic
    replacement and exception logging; this function only logs the count upon success to confirm that traceability
    information has been persisted.
    """
    try:
        saved = task_artifacts.patch_script_data(
            task_id,
            material_sources=material_sources,
        )
        if saved:
            logger.info(
                f"saved material source records: "
                f"task_id={task_id}, count={len(material_sources)}"
            )
    except Exception as exc:
        # task_artifacts is designed with failure fallback, but this final isolation layer is retained
        # to prevent future implementation changes or directory resolution errors from affecting the download return value.
        logger.warning(
            "failed to persist material source records: "
            f"task_id={task_id}, error={type(exc).__name__}, detail={exc}"
        )


def _get_tls_verify() -> bool:
    # TLS certificate verification is enabled by default to prevent MITM tampering during search and download.
    # Only in scenarios strictly requiring it (such as enterprise proxies or self-signed certs), users may
    # temporarily disable it by explicitly setting `tls_verify = false` in `config.toml`.
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")

    if not tls_verify:
        logger.warning(
            "TLS certificate verification is disabled by config.app.tls_verify=false. "
            "Only use this in trusted proxy environments."
        )

    return bool(tls_verify)


def get_api_key(cfg_key: str):
    api_keys = config.app.get(cfg_key)
    if not api_keys:
        raise ValueError(
            f"\n\n##### {cfg_key} is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )

    # if only one key is provided, return it
    if isinstance(api_keys, str):
        return api_keys

    global _api_key_counter
    with _api_key_lock:
        _api_key_counter += 1
        return api_keys[_api_key_counter % len(api_keys)]


def _redact_secret(message: str, secret: str) -> str:
    """
    Perform minimal redaction on exception text before logging.

    Requests connection exceptions may include the full request URL, while Pixabay API keys are passed
    via query parameters. Both raw and URL-encoded values are redacted here, retaining network error details
    for troubleshooting while preventing secrets from entering log files.
    """
    safe_message = str(message)
    if not secret:
        return safe_message

    safe_message = safe_message.replace(secret, "***")
    encoded_secret = quote_plus(secret)
    if encoded_secret != secret:
        safe_message = safe_message.replace(encoded_secret, "***")
    return safe_message


def _redact_request_error(error: Exception, *secrets: str) -> str:
    """
    Retain troubleshooting context for network exceptions while removing API keys and proxy credentials.

    Logging only the exception class loses critical context such as DNS, certificate, or timeout details;
    logging the raw exception directly may echo the full request URL. A unified entry point applies consistent
    redaction rules across all three stock providers.
    """
    safe_message = str(error)
    for secret in secrets:
        safe_message = _redact_secret(safe_message, str(secret or ""))
    for proxy_url in config.proxy.values():
        safe_message = _redact_secret(safe_message, str(proxy_url))
    return safe_message


def _is_cloudflare_challenge(response: requests.Response) -> bool:
    """
    Identify HTML Challenges returned by Cloudflare rather than treating them as Pixabay JSON.

    Cloudflare typically sets `cf-mitigated: challenge`; some deployments only return HTML containing
    "Just a moment" or challenge-platform, so content feature matching is retained as fallback.
    The response body is evaluated only in memory and never logged to avoid polluting logs with large HTML dumps.
    """
    headers = getattr(response, "headers", {}) or {}
    if str(headers.get("cf-mitigated", "")).lower() == "challenge":
        return True

    content_type = str(headers.get("content-type", "")).lower()
    if "text/html" not in content_type:
        return False

    body = str(getattr(response, "text", "")).lower()
    return "just a moment" in body or "/cdn-cgi/challenge-platform/" in body


def _matches_video_aspect(
    width: Any,
    height: Any,
    video_aspect: VideoAspect,
    *,
    is_vertical: Any = None,
) -> bool:
    """
    Determine whether remote material matches the target video aspect ratio.

    Response schemas differ across Pexels, Pixabay, and Coverr, so width and height are checked first for reliability;
    when Coverr legacy responses lack dimensions, the explicit ``is_vertical`` boolean is used as a fallback.
    Materials with indeterminable orientation are skipped to prevent landscape clips from introducing black bars into portrait videos.
    """
    aspect = VideoAspect(video_aspect)
    try:
        normalized_width = int(float(width))
        normalized_height = int(float(height))
    except (OverflowError, TypeError, ValueError):
        normalized_width = 0
        normalized_height = 0

    if normalized_width > 0 and normalized_height > 0:
        if aspect == VideoAspect.portrait:
            return normalized_height > normalized_width
        if aspect == VideoAspect.landscape:
            return normalized_width > normalized_height
        return normalized_width == normalized_height

    if isinstance(is_vertical, bool) and aspect != VideoAspect.square:
        return is_vertical == (aspect == VideoAspect.portrait)
    return False


def _filter_materials_by_aspect(
    items: List[MaterialInfo],
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """
    Re-validate orientation on cached results.

    Material search caches are retained up to 24 hours; caches created before upgrades may contain mismatched orientations.
    Filtering at the unified cache entry point applies fixes immediately and protects against third-party providers or stale caches
    missing remote filtering. Legacy entries whose rendition dimensions cannot be read are treated as unverified and skipped.
    """
    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.square:
        # Pixabay and Coverr rarely provide native square materials. Square output preserves existing behavior,
        # accepting available candidates and delegating cropping to video synthesis to avoid empty results for 1:1 tasks.
        return list(items)

    filtered_items = []
    for item in items:
        source_info = item.source_info if isinstance(item.source_info, dict) else {}
        rendition = source_info.get("rendition")
        rendition = rendition if isinstance(rendition, dict) else {}
        if _matches_video_aspect(
            rendition.get("width"),
            rendition.get("height"),
            aspect,
        ):
            filtered_items.append(item)
    return filtered_items


def search_videos_pexels(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("pexels_api_keys")
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    # Build URL
    params = {"query": search_term, "per_page": 20}
    if aspect != VideoAspect.square:
        params["orientation"] = aspect.name
    query_url = f"https://api.pexels.com/v1/videos/search?{urlencode(params)}"
    logger.info(f"searching videos on pexels: term={search_term!r}")

    try:
        r = trace_http_call(
            "GET", requests.get,
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        response = r.json()
        video_items = []
        if not isinstance(response, dict) or not isinstance(
            response.get("videos"), list
        ):
            logger.error("pexels video search returned an unsupported response")
            return video_items
        videos = response["videos"]
        # loop through each video in the result
        for v in videos:
            if not isinstance(v, dict):
                continue
            duration = v.get("duration")
            # check if video has desired minimum duration
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < minimum_duration
            ):
                continue
            video_files = v.get("video_files")
            if not isinstance(video_files, list):
                continue
            # Prefer the smallest rendition that can fill the output canvas.
            # Square output can crop either orientation, as other providers do.
            renditions = []
            for video in video_files:
                if not isinstance(video, dict):
                    continue
                try:
                    w = int(video.get("width"))
                    h = int(video.get("height"))
                except (OverflowError, TypeError, ValueError):
                    continue
                video_url = video.get("link")
                if not isinstance(video_url, str) or not video_url:
                    continue
                if (
                    (aspect == VideoAspect.square or _matches_video_aspect(w, h, aspect))
                    and w >= video_width
                    and h >= video_height
                ):
                    renditions.append((w * h, video, w, h, video_url))
            if renditions:
                _, video, w, h, video_url = min(renditions, key=lambda candidate: candidate[0])
                item = MaterialInfo()
                item.provider = "pexels"
                item.url = video_url
                item.duration = duration
                item.source_info = {
                    "provider": "pexels",
                    "search_term": search_term,
                    "asset_id": (
                        str(v.get("id")) if v.get("id") is not None else None
                    ),
                    "source_page": _safe_public_url(v.get("url")),
                    "creator": _creator_info(v.get("user")),
                    "rendition": {
                        "id": (
                            str(video.get("id"))
                            if video.get("id") is not None
                            else None
                        ),
                        "width": w,
                        "height": h,
                    },
                }
                video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "pexels video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


def search_videos_pixabay(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)

    video_width, video_height = aspect.to_resolution()

    api_key = get_api_key("pixabay_api_keys")
    # Build URL
    params = {
        "q": search_term,
        "video_type": "all",  # Accepted values: "all", "film", "animation"
        "per_page": 50,
        "key": api_key,
    }
    query_url = f"https://pixabay.com/api/videos/?{urlencode(params)}"
    logger.info(
        f"searching videos on pixabay: term={search_term!r}, "
        f"proxy_enabled={bool(config.proxy)}"
    )

    try:
        r = trace_http_call(
            "GET", requests.get,
            query_url, proxies=config.proxy, verify=_get_tls_verify(), timeout=(30, 60)
        )
        status_code = int(getattr(r, "status_code", 200))
        headers = getattr(r, "headers", {}) or {}
        content_type = str(headers.get("content-type", ""))
        retry_after = headers.get("retry-after")
        cf_ray = headers.get("cf-ray")

        if _is_cloudflare_challenge(r):
            logger.error(
                "pixabay search was blocked by a Cloudflare challenge: "
                f"status={status_code}, cf_ray={cf_ray or 'unknown'}. "
                "Check the server network or proxy, or use Pexels/Coverr instead."
            )
            return []

        if status_code == 429:
            logger.error(
                "pixabay API rate limit exceeded: "
                f"status=429, retry_after={retry_after or 'unknown'}"
            )
            return []

        if status_code >= 400:
            logger.error(
                "pixabay search request failed: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        try:
            response = r.json()
        except ValueError:
            logger.error(
                "pixabay returned an unexpected non-JSON response: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        video_items = []
        if not isinstance(response, dict) or not isinstance(
            response.get("hits"), list
        ):
            logger.error("pixabay video search returned an unsupported response")
            return video_items
        videos = response["hits"]
        # loop through each video in the result
        for v in videos:
            if not isinstance(v, dict):
                continue
            duration = v.get("duration")
            # check if video has desired minimum duration
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < minimum_duration
            ):
                continue
            video_files = v.get("videos")
            if not isinstance(video_files, dict):
                continue
            # loop through each url to determine the best quality
            for video_type, video in video_files.items():
                if not isinstance(video, dict):
                    continue
                try:
                    w = int(video["width"])
                    h = int(video["height"])
                except (KeyError, OverflowError, TypeError, ValueError):
                    continue
                video_url = video.get("url")
                if not isinstance(video_url, str) or not video_url:
                    continue
                # Pixabay rarely returns native square videos; 1:1 output continues to accept candidates
                # meeting resolution thresholds for cropping during synthesis. Landscape and portrait must match strictly.
                orientation_matches = aspect == VideoAspect.square or (
                    _matches_video_aspect(w, h, aspect)
                )
                if orientation_matches and w >= video_width and h >= video_height:
                    item = MaterialInfo()
                    item.provider = "pixabay"
                    item.url = video_url
                    item.duration = duration
                    item.source_info = {
                        "provider": "pixabay",
                        "search_term": search_term,
                        "asset_id": (
                            str(v.get("id")) if v.get("id") is not None else None
                        ),
                        "source_page": _safe_public_url(v.get("pageURL")),
                        "creator": _creator_info(
                            {
                                "id": v.get("user_id"),
                                "name": v.get("user"),
                            }
                        ),
                        "rendition": {
                            "id": video_type,
                            "width": w,
                            "height": video.get("height"),
                        },
                    }
                    video_items.append(item)
                    break
        return video_items
    except Exception as e:
        error_message = _redact_request_error(e, api_key)
        logger.error(
            "pixabay search request failed: "
            f"error={type(e).__name__}, detail={error_message}"
        )

    return []


def search_videos_coverr(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    """
    Coverr (https://coverr.co) - free HD/4K stock videos,
    subject to Coverr license terms (https://coverr.co/license).

    Coverr API notes (based on official docs at api.coverr.co/docs/):
      - Authentication: Authorization: Bearer <api_key>
      - Search endpoint: GET /videos?query=..., response schema {"hits": [...], ...}
      - Append ?urls=true to return direct mp4 URLs in the search response
      - URL is a signed JWT (bound to API key, no expiration)
      - Coverr supports filtering orientation via filter=is_vertical:true/false;
        responses are still validated locally via max_width/max_height or is_vertical
      - The duration field may appear as either number or string; both are accepted

    This function uses the urls.mp4_download field as download URL — according to Coverr official documentation
    (https://api.coverr.co/docs/videos/#download-a-video),
    issuing a GET on this URL is treated by Coverr as a legitimate download event for analytics,
    without requiring an additional call to PATCH /videos/:id/stats/downloads.
    """
    aspect = VideoAspect(video_aspect)
    api_key = get_api_key("coverr_api_keys")
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "query": search_term,
        "page_size": 20,
        "urls": "true",
        "sort": "popular",
    }
    # Server-side orientation filtering returns matching assets directly from the full search index, avoiding empty portrait
    # candidates caused by fetching top hits first and filtering locally. Square material lacks a boolean filter and relies on local dimension checks.
    if aspect == VideoAspect.portrait:
        params["filter"] = "is_vertical:true"
    elif aspect == VideoAspect.landscape:
        params["filter"] = "is_vertical:false"
    query_url = f"https://api.coverr.co/videos?{urlencode(params)}"
    logger.info(f"searching videos on coverr: term={search_term!r}")

    try:
        r = trace_http_call(
            "GET", requests.get,
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        response = r.json()
        video_items: List[MaterialInfo] = []

        if not isinstance(response, dict) or not isinstance(
            response.get("hits"), list
        ):
            logger.error("coverr video search returned an unsupported response")
            return video_items

        for v in response["hits"]:
            if not isinstance(v, dict):
                continue
            # duration may be number(11.625) or string("10.500000") in different responses
            try:
                duration = int(float(v.get("duration") or 0))
            except (OverflowError, TypeError, ValueError):
                continue
            if duration < minimum_duration:
                continue

            video_id = v.get("id")
            urls = v.get("urls")
            if not isinstance(urls, dict):
                continue
            mp4_download_url = urls.get("mp4_download")
            if (
                not video_id
                or not isinstance(mp4_download_url, str)
                or not mp4_download_url
            ):
                continue
            if aspect != VideoAspect.square and not _matches_video_aspect(
                v.get("max_width"),
                v.get("max_height"),
                aspect,
                is_vertical=v.get("is_vertical"),
            ):
                continue

            item = MaterialInfo()
            item.provider = "coverr"
            item.url = mp4_download_url
            item.duration = duration
            item.source_info = {
                "provider": "coverr",
                "search_term": search_term,
                "asset_id": str(video_id),
                "source_page": _safe_public_url(v.get("canonical_url") or v.get("url")),
                "creator": _creator_info(v.get("creator") or v.get("author")),
                "rendition": {
                    "id": "mp4_download",
                    "width": v.get("max_width"),
                    "height": v.get("max_height"),
                },
            }
            video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "coverr video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


# WaveSpeed AI (https://wavespeed.ai) generates materials directly from script keywords via text-to-video models,
# sharing the MaterialInfo schema and downstream download/editing pipeline with the three stock sources.
WAVESPEED_API_BASE_URL = "https://api.wavespeed.ai/api/v3"
WAVESPEED_DEFAULT_T2V_MODEL = "bytedance/seedance-2.0-fast/text-to-video"
WAVESPEED_POLL_INTERVAL_SECONDS = 2.0
WAVESPEED_RUN_TIMEOUT_SECONDS = 600.0
# Default model bytedance/seedance-2.0-fast/text-to-video only accepts 4-15 seconds; out-of-range requests
# are rejected by the API. The WebUI default segment duration is 3 seconds, so values must be clamped to the supported
# range before submission; excess duration is trimmed by the editing pipeline to match segment duration.
WAVESPEED_MIN_DURATION_SECONDS = 4
WAVESPEED_MAX_DURATION_SECONDS = 15
# The three failure states have different semantics (model error / user cancellation / platform timeout), but for the material pipeline
# all mean no output was generated for the keyword; treat uniformly as empty results so upper layers skip and proceed.
WAVESPEED_FAILURE_STATUSES = frozenset({"failed", "cancelled", "timeout"})
# Aligned with official WaveSpeed Python SDK / n8n nodes: 429 and 5xx are transient errors eligible
# for bounded backoff retry; 4xx represents explicit client errors that fail fast.
WAVESPEED_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Maximum consecutive transient failures allowed during a single poll. An unlucky GET should not lose track of a billed task.
WAVESPEED_MAX_POLL_RETRIES = 5
# Linear backoff base; the n-th retry waits base * n seconds.
WAVESPEED_RETRY_BASE_SECONDS = 1.0
# Retry attempts on the same signed URL when asset download fails. Material is already paid and generated; retry
# the original URL first rather than submitting another paid generation task due to download jitter.
WAVESPEED_MAX_DOWNLOAD_RETRIES = 2


class WaveSpeedUnconfirmedTaskError(RuntimeError):
    """
    A paid generation task was submitted, but its final status cannot be confirmed locally.

    This error is never equivalent to "task failed, safe to retry": the remote task may still be running or completed
    and billed. The material pipeline must halt immediately, submit no further paid tasks for remaining keywords,
    and propagate the submitted prediction id into the task state for manual retrieval.
    """

    def __init__(self, message: str, prediction_id: str = ""):
        super().__init__(message)
        self.prediction_id = prediction_id


class WaveSpeedDownloadError(RuntimeError):
    """A paid prediction completed, but its video could not be saved locally."""

    def __init__(self, message: str, prediction_id: str = ""):
        super().__init__(message)
        self.prediction_id = prediction_id


def _wavespeed_status_code(response: Any) -> int:
    """Read response status code; treat as 200 when test doubles or exception objects lack this field."""
    try:
        return int(getattr(response, "status_code", 200))
    except (TypeError, ValueError):
        return 200


def _is_wavespeed_retryable_error(error: Exception) -> bool:
    """
    Determine whether a polling exception warrants a retry.

    Network errors like connection or timeout lack status codes and are treated as transient failures;
    responses with status codes are only retried on 429 and 5xx, matching the official SDK retry behavior.
    """
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
    if response is not None:
        return _wavespeed_status_code(response) in WAVESPEED_RETRYABLE_STATUS_CODES
    return False


def _wavespeed_duration_bounds() -> tuple[int, int]:
    """
    Return generation duration bounds (in seconds) supported by the current model.

    The default range corresponds to the default Seedance model; when switching to other text-to-video models,
    users can adjust bounds in the configuration. Invalid configurations fall back to defaults with min <= max guaranteed,
    preventing user input from producing doomed remote requests.
    """

    def read_bound(key: str, fallback: int) -> int:
        try:
            value = int(config.app.get(key, fallback))
        except (TypeError, ValueError):
            return fallback
        return value if value >= 1 else fallback

    min_duration = read_bound("wavespeed_min_duration", WAVESPEED_MIN_DURATION_SECONDS)
    max_duration = read_bound("wavespeed_max_duration", WAVESPEED_MAX_DURATION_SECONDS)
    return min_duration, max(max_duration, min_duration)


def generate_videos_wavespeed(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    """
    Generate video material for a script keyword using the WaveSpeed text-to-video model.

    Maintains the same signature and empty-list failure convention as stock search_videos_* sources,
    allowing seamless integration into the common ``download_videos`` download and duration accounting pipeline.
    In the generation context, ``minimum_duration`` represents the target segment duration (seconds).
    """
    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("wavespeed_api_keys")
    model_id = (
        str(
            config.app.get("wavespeed_text_to_video_model", "")
            or WAVESPEED_DEFAULT_T2V_MODEL
        )
        .strip()
        .strip("/")
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    requested_duration = max(int(minimum_duration), 1)
    min_duration, max_duration = _wavespeed_duration_bounds()
    duration = min(max(requested_duration, min_duration), max_duration)
    if duration != requested_duration:
        # Generating longer than requested does not affect the final video: the editing pipeline clips to segment duration;
        # generating shorter only happens when requests exceed the model ceiling, where clamping to the ceiling is necessary.
        logger.info(
            f"wavespeed clip duration clamped to model-supported range: "
            f"requested={requested_duration}s, using={duration}s "
            f"(supported {min_duration}-{max_duration}s)"
        )
    payload = {
        "prompt": search_term,
        "aspect_ratio": aspect.value,
        "duration": duration,
    }
    logger.info(
        f"generating video on wavespeed: model={model_id}, "
        f"term={search_term!r}, duration={duration}s"
    )

    # Never auto-retry POST submissions: the request may have already created a paid task remotely; re-sending
    # causes duplicate generation and double billing (matching official SDK submission strategy).
    try:
        submit_response = trace_http_call(
            "POST", requests.post,
            f"{WAVESPEED_API_BASE_URL}/{model_id}",
            json=payload,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
    except Exception as e:
        # Not receiving a response does not mean the task was not created. Status is indeterminate; halt the entire
        # generation flow rather than submitting new paid tasks for subsequent keywords.
        raise WaveSpeedUnconfirmedTaskError(
            "wavespeed submission did not return a response, the task may "
            "already exist remotely: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        ) from e

    submit_status = _wavespeed_status_code(submit_response)
    if submit_status >= 500:
        # 5xx may occur after task creation; impossible to confirm whether billing occurred.
        raise WaveSpeedUnconfirmedTaskError(
            f"wavespeed submission failed with HTTP {submit_status}, "
            "the task may already exist remotely"
        )
    try:
        submit_body = submit_response.json()
    except Exception as e:
        if 400 <= submit_status < 500:
            # Gateway or auth layer 4xx may return plain text/HTML. 4xx is still an explicit refusal with no remote
            # task created, so do not misreport as "potentially billed" to abort the entire task.
            logger.error(
                "wavespeed video generation request rejected: "
                f"http_status={submit_status}, unreadable body: {type(e).__name__}"
            )
            return []
        raise WaveSpeedUnconfirmedTaskError(
            "wavespeed submission returned an unreadable response, the task "
            f"may already exist remotely: error={type(e).__name__}"
        ) from e

    submit_data = submit_body.get("data") if isinstance(submit_body, dict) else None
    if not isinstance(submit_body, dict) or submit_body.get("code") != 200:
        # 4xx and business error codes are explicit rejections without remote task creation or double-billing risk;
        # return empty results according to existing material source conventions and continue.
        logger.error(
            "wavespeed video generation request rejected: "
            f"http_status={submit_status}, "
            f"code={submit_body.get('code') if isinstance(submit_body, dict) else None}, "
            f"detail={_redact_secret(str((submit_body or {}).get('message') or ''), api_key)}"
        )
        return []
    prediction_id = (
        str(submit_data.get("id") or "") if isinstance(submit_data, dict) else ""
    )
    if not prediction_id:
        # Submission was accepted but no ID returned: task may exist untracked, cannot place further orders.
        raise WaveSpeedUnconfirmedTaskError(
            "wavespeed accepted the submission without returning a prediction id"
        )
    # Successful submission carries remote billing side effects; log task ID immediately so that
    # even if subsequent polling fails, users can retrieve assets from the WaveSpeed console using the ID.
    logger.info(f"wavespeed prediction created: id={prediction_id}")

    result_data = _wait_for_wavespeed_prediction(
        prediction_id=prediction_id,
        headers=headers,
        api_key=api_key,
    )
    if result_data is None:
        return []

    try:
        video_items = []
        outputs = result_data.get("outputs")
        for output in outputs if isinstance(outputs, list) else []:
            # Asset URL is a signed temporary download URL that must be preserved whole (query params intact),
            # so it is not written to source_info and is used only for immediate download.
            if not isinstance(output, str) or not output.startswith(
                ("http://", "https://")
            ):
                continue
            item = MaterialInfo()
            item.provider = "wavespeed"
            item.url = output
            item.duration = duration
            item.source_info = {
                "provider": "wavespeed",
                "search_term": search_term,
                "asset_id": prediction_id,
                "rendition": {
                    "id": None,
                    "width": video_width,
                    "height": video_height,
                },
            }
            video_items.append(item)
        if not video_items:
            logger.error(
                "wavespeed prediction completed without downloadable outputs: "
                f"id={prediction_id}"
            )
        return video_items
    except Exception as e:
        # Asset has been generated and billed; exceptions here stem only from local parsing. Log and return
        # empty result to skip this segment; the task status itself is deterministic so subsequent segments may proceed.
        logger.error(
            "wavespeed output parsing failed: "
            f"id={prediction_id}, error={type(e).__name__}, "
            f"detail={_redact_request_error(e, api_key)}"
        )

    return []


def _wait_for_wavespeed_prediction(
    *,
    prediction_id: str,
    headers: dict,
    api_key: str,
) -> dict | None:
    """
    Poll the same prediction id until a definitive result is reached.

    Returns ``completed`` data; returns None on explicit remote failures (failed / cancelled / timeout),
    indicating the task has ended and subsequent segments can safely proceed. Transient failures retry the same ID
    with linear backoff without ever re-submitting; if status remains unconfirmed, raises
    :class:`WaveSpeedUnconfirmedTaskError` for the caller to terminate the entire generation flow.
    """
    deadline = time.monotonic() + WAVESPEED_RUN_TIMEOUT_SECONDS
    consecutive_failures = 0
    while True:
        try:
            response = trace_http_call(
                "GET", requests.get,
                f"{WAVESPEED_API_BASE_URL}/predictions/{prediction_id}/result",
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 60),
            )
            status_code = _wavespeed_status_code(response)
            if status_code in WAVESPEED_RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"HTTP {status_code}", response=response
                )
            result_body = response.json()
            result_data = (
                result_body.get("data") if isinstance(result_body, dict) else None
            )
            if not isinstance(result_body, dict) or result_body.get("code") != 200:
                # When polling is explicitly rejected (e.g. 4xx), task status remains unknown: the task was submitted
                # but local lookup failed, so no further paid tasks should be submitted.
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction status is unknown: "
                    f"http_status={status_code}, "
                    f"code={result_body.get('code') if isinstance(result_body, dict) else None}, "
                    f"detail={_redact_secret(str((result_body or {}).get('message') or ''), api_key)}",
                    prediction_id=prediction_id,
                )
            if not isinstance(result_data, dict):
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction result payload is malformed",
                    prediction_id=prediction_id,
                )
        except WaveSpeedUnconfirmedTaskError:
            raise
        except Exception as e:
            if not _is_wavespeed_retryable_error(e):
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction polling failed and the task state is "
                    f"unknown: error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, api_key)}",
                    prediction_id=prediction_id,
                ) from e
            consecutive_failures += 1
            if consecutive_failures > WAVESPEED_MAX_POLL_RETRIES:
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction polling failed after "
                    f"{WAVESPEED_MAX_POLL_RETRIES + 1} attempts, the task may "
                    "still be running remotely: "
                    f"error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, api_key)}",
                    prediction_id=prediction_id,
                ) from e
            delay = WAVESPEED_RETRY_BASE_SECONDS * consecutive_failures
            logger.warning(
                "wavespeed prediction polling hit a transient error, retry the "
                f"same task: id={prediction_id}, "
                f"attempt={consecutive_failures}/{WAVESPEED_MAX_POLL_RETRIES}, "
                f"error={type(e).__name__}, retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        # Reset counter on any valid response; retry quota is consumed only by consecutive failures.
        consecutive_failures = 0
        status = str(result_data.get("status") or "")
        if status == "completed":
            return result_data
        if status in WAVESPEED_FAILURE_STATUSES:
            logger.error(
                "wavespeed prediction did not produce a video: "
                f"id={prediction_id}, status={status}, "
                f"detail={_redact_secret(str(result_data.get('error') or ''), api_key)}"
            )
            return None
        if time.monotonic() > deadline:
            # Remote task is still running; local final status cannot be confirmed, so further ordering must cease.
            raise WaveSpeedUnconfirmedTaskError(
                f"wavespeed prediction is still {status or 'pending'} after "
                f"{WAVESPEED_RUN_TIMEOUT_SECONDS:.0f}s of local waiting",
                prediction_id=prediction_id,
            )
        time.sleep(WAVESPEED_POLL_INTERVAL_SECONDS)


def _save_generated_video_with_retry(
    video_url: str, save_dir: str, provider: str
) -> str:
    """
    Download paid and generated assets, retrying the same URL first on failure.

    Re-generating remotely costs another fee, so download jitter must be retried on the original URL with
    bounded backoff first; once retries are exhausted, the caller reports a recoverable paid task failure.
    """
    for attempt in range(WAVESPEED_MAX_DOWNLOAD_RETRIES + 1):
        try:
            saved_video_path = save_video(video_url=video_url, save_dir=save_dir)
            if saved_video_path:
                return saved_video_path
            failure_detail = "empty result"
        except Exception as e:
            failure_detail = (
                f"error={type(e).__name__}, "
                f"detail={_redact_request_error(e, video_url)}"
            )
        if attempt >= WAVESPEED_MAX_DOWNLOAD_RETRIES:
            break
        delay = WAVESPEED_RETRY_BASE_SECONDS * (attempt + 1)
        logger.warning(
            "failed to download generated video, retry the same url: "
            f"provider={provider}, "
            f"attempt={attempt + 1}/{WAVESPEED_MAX_DOWNLOAD_RETRIES}, "
            f"{failure_detail}, retry_in={delay:.1f}s"
        )
        time.sleep(delay)
    logger.error(
        "failed to download generated video after "
        f"{WAVESPEED_MAX_DOWNLOAD_RETRIES + 1} attempts: "
        f"provider={provider}, {failure_detail}"
    )
    return ""


def _get_downloaded_video_duration(video_path: str) -> float:
    """Read the usable duration from the downloaded media file."""
    clip = None
    try:
        clip = VideoFileClip(video_path)
        duration = float(clip.duration or 0)
    except Exception as exc:
        raise ValueError(
            f"downloaded video duration could not be measured: {video_path}"
        ) from exc
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception as close_error:
                logger.warning(
                    "failed to close downloaded video after duration probe: "
                    f"path={video_path}, error={type(close_error).__name__}, "
                    f"detail={close_error}"
                )
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            f"downloaded video duration is not positive and finite: {video_path}"
        )
    return duration


# Requests stream download emits no logs during transfer. Over slow networks, downloading a 4K asset can take minutes,
# making tasks look frozen. Log downloaded size and speed when exceeding this interval; small files completed within
# the interval omit extra logs to avoid log spam.
_DOWNLOAD_HEARTBEAT_SECONDS = 15.0
_BYTES_PER_MEGABYTE = 1024 * 1024


def _describe_download_progress(
    downloaded_bytes: int, declared_size: int, elapsed_seconds: float
) -> str:
    """Format download progress into a log message; omit percentage if server does not declare content length."""
    downloaded_mb = downloaded_bytes / _BYTES_PER_MEGABYTE
    speed = downloaded_mb / elapsed_seconds if elapsed_seconds > 0 else 0.0
    if declared_size > 0:
        percent = min(100, round(downloaded_bytes * 100 / declared_size))
        return (
            f"{downloaded_mb:.1f} of {declared_size / _BYTES_PER_MEGABYTE:.1f} MB "
            f"({percent}%), {speed:.2f} MB/s"
        )
    return f"{downloaded_mb:.1f} MB, {speed:.2f} MB/s"


def save_video(video_url: str, save_dir: str = "") -> str:
    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")

    os.makedirs(save_dir, exist_ok=True)

    # Query parameters can identify the asset itself (for example,
    # /download?file_id=123). Dropping the query makes unrelated paid videos
    # share one cache entry and silently reuse the first scene. Fragments are
    # not sent in HTTP requests, so they do not affect the downloaded bytes.
    url_hash = utils.md5(video_url.split("#", 1)[0])
    video_id = f"vid-{url_hash}"
    video_path = f"{save_dir}/{video_id}.mp4"

    # if video already exists, return the path
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        logger.info(f"video already exists: {video_path}")
        return video_path

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    # A nonempty file is treated as a cache hit above. Keep the final path
    # unpublished until the complete download has passed media validation, so a
    # failed download cannot poison this or another concurrent task's cache.
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{video_id}-",
            suffix=".mp4",
            dir=save_dir,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            with trace_http_call(
                "GET", requests.get,
                video_url,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(60, 240),
                stream=True,
            ) as response:
                response.raise_for_status()
                headers = getattr(response, "headers", {}) or {}
                try:
                    declared_size = int(headers.get("Content-Length", ""))
                except (TypeError, ValueError):
                    declared_size = 0
                if declared_size > MAX_VIDEO_DOWNLOAD_BYTES:
                    raise ValueError("video download exceeds 512 MB limit")

                downloaded_bytes = 0
                started_at = time.monotonic()
                last_report_at = started_at
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > MAX_VIDEO_DOWNLOAD_BYTES:
                            raise ValueError("video download exceeds 512 MB limit")
                        temp_file.write(chunk)
                        now = time.monotonic()
                        if now - last_report_at >= _DOWNLOAD_HEARTBEAT_SECONDS:
                            last_report_at = now
                            # Log cache filename only: download URL may contain signatures or keys.
                            logger.info(
                                f"downloading video {video_id}.mp4: "
                                + _describe_download_progress(
                                    downloaded_bytes,
                                    declared_size,
                                    now - started_at,
                                )
                            )

        if os.path.getsize(temp_path) == 0:
            return ""

        clip = None
        try:
            clip = VideoFileClip(temp_path)
            duration = clip.duration
            fps = clip.fps
            # Container duration and an initial frame do not prove the rest of
            # a faststart MP4 arrived. Validate every video packet before a
            # nonempty cache entry becomes reusable by later tasks.
            material_upload._validate_video(temp_path)
            if not (duration > 0 and fps > 0):
                logger.warning(f"invalid video file: {temp_path} => invalid duration or fps")
                return ""
        except Exception as e:
            logger.warning(f"invalid video file: {temp_path} => {str(e)}")
            return ""
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception as close_error:
                    logger.warning(
                        f"failed to close video clip: {temp_path}, error: {str(close_error)}"
                    )

        os.replace(temp_path, video_path)
        temp_path = ""
        return video_path
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
            except OSError as remove_error:
                logger.warning(
                    f"failed to remove temporary video file: {temp_path}, "
                    f"error: {str(remove_error)}"
                )


# OpenAI-compatible text-to-image (Issue #1274) generates image assets for script keywords via /images/generations,
# targeting either local ComfyUI/SD gateways or OpenAI-compatible relay services. Generated images are immediately
# rendered into "slow zoom" mp4 clips matching local image behavior, completely transparent to downstream editing.
OPENAI_IMAGE_ENDPOINT_PATH = "images/generations"
# Official OpenAI image API only accepts model-defined sizes, not arbitrary video resolutions (e.g. 1080x1920).
# When openai_image_size is empty, use compatible defaults below by aspect ratio; local gateways can explicitly override.
OPENAI_IMAGE_DEFAULT_SIZES = {
    VideoAspect.portrait: "1024x1536",
    VideoAspect.landscape: "1536x1024",
    VideoAspect.square: "1024x1024",
}
# Same retry policy as WaveSpeed: 429 and 5xx are transient errors eligible for bounded backoff retries.
OPENAI_IMAGE_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# 401/403 indicates explicit rejection of the current key. get_api_key rotates keys on each call, so multi-key setups
# automatically switch keys on retry; single-key setups fail fast to avoid futile attempts.
OPENAI_IMAGE_KEY_ERROR_STATUS_CODES = frozenset({401, 403})
OPENAI_IMAGE_MAX_ATTEMPTS = 3
OPENAI_IMAGE_MAX_BYTES = 25 * 1024 * 1024
OPENAI_IMAGE_MAX_PIXELS = 50_000_000
# Sequential image generation + linear backoff, compatible with common rate-limit recovery windows on relay services.
OPENAI_IMAGE_RETRY_BACKOFF_SECONDS = (5, 15, 30)
# Synchronous generation APIs may take tens of seconds to return images; provide generous read timeout.
OPENAI_IMAGE_REQUEST_TIMEOUT = (30, 300)
# Download retry after image has been billed per unit: retry original URL first rather than re-generating the image.
OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS = 3
OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS = 2


def is_openai_image_enabled(app_config: dict | None = None) -> bool:
    """
    Determine whether the OpenAI-compatible text-to-image source has minimal valid configuration.

    API Key may be empty: fully local ComfyUI/SD gateways typically require no auth, and omitting it sends no Authorization header.
    Used by task pre-checks and WebUI to intercept misconfigured tasks before consuming LLM and TTS credits.
    """
    app_config = config.app if app_config is None else app_config
    return bool(
        str(app_config.get("openai_image_base_url", "") or "").strip()
        and str(app_config.get("openai_image_model", "") or "").strip()
    )


def _openai_image_endpoint() -> tuple[str, str]:
    """
    Read text-to-image endpoint and model name; raise configuration guidance error when missing.
    """
    base_url = (
        str(config.app.get("openai_image_base_url", "") or "").strip().rstrip("/")
    )
    model = str(config.app.get("openai_image_model", "") or "").strip()
    if not base_url:
        raise ValueError(
            "\n\n##### openai_image_base_url is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )
    if not model:
        raise ValueError(
            "\n\n##### openai_image_model is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )
    return f"{base_url}/{OPENAI_IMAGE_ENDPOINT_PATH}", model


def _openai_image_size(video_aspect: VideoAspect) -> str:
    """
    Resolve requested image size.

    Official OpenAI endpoints only accept model-prescribed sizes (e.g. 1024x1536); passing video resolutions directly
    (e.g. 1080x1920) returns 400. Defaults to compatible sizes by aspect ratio; ``openai_image_size`` can explicitly
    override for local gateways (such as SD WebUI) supporting arbitrary resolutions.
    """
    configured = str(config.app.get("openai_image_size", "") or "").strip()
    if configured:
        return configured
    return OPENAI_IMAGE_DEFAULT_SIZES.get(VideoAspect(video_aspect), "1024x1024")


def _openai_image_prompt(search_term: str) -> str:
    """
    Wrap script keywords into the final prompt.

    Optional configuration ``openai_image_prompt_template`` supports a ``{term}`` placeholder
    to append uniform style modifiers (quality, composition, camera language) for better text-visual match:

    .. code-block:: toml

        openai_image_prompt_template = "cinematic photo of {term}, photorealistic"

    When empty or lacking the placeholder, falls back to raw keyword matching legacy behavior.
    Placeholder replacement failures (e.g. malformed template syntax) also fall back to raw keyword,
    preventing configuration errors from aborting the generation task.
    """
    template = str(config.app.get("openai_image_prompt_template", "") or "").strip()
    if not template or "{term}" not in template:
        return search_term
    try:
        return template.replace("{term}", search_term)
    except Exception:
        return search_term


def _response_json_safely(response: Any) -> Any:
    """Read response JSON safely; return None when test doubles or error responses fail to parse."""
    try:
        return response.json()
    except Exception:
        return None


def _openai_image_response_message(body: Any) -> str:
    """
    Extract a human-readable error description from an OpenAI-compatible response.

    Standard schema is ``{"error": {"message": ...}}``, but relay services often degrade to
    ``{"message": ...}`` or a raw string. Returns an empty string if neither is found, letting callers
    fall back to response text.
    """
    if not isinstance(body, dict):
        return str(body or "")[:300]
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")[:300]
    if error is not None:
        return str(error)[:300]
    return str(body.get("message") or "")[:300]


def _openai_image_http_failure(response: Any, status: int, api_key: str) -> str:
    """Format an HTTP error response into a redacted, log-friendly description."""
    message = _openai_image_response_message(_response_json_safely(response))
    if not message:
        message = str(getattr(response, "text", "") or "")[:300]
    return f"HTTP {status}: {_redact_secret(message, api_key)}"


def _openai_image_download_bytes(
    image_url: str,
    api_key: str,
) -> tuple[bytes | None, str]:
    """
    Download a temporary URL for an already generated image.

    The image is already billed per unit, so retrying the original URL on failure takes precedence over re-generation
    to avoid double paying for the same asset. Raises on exhausted retries to stop outer loops from purchasing subsequent images.
    """
    failure_detail = "no download attempt was made"
    for attempt in range(1, OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS + 1):
        response = None
        try:
            response = trace_http_call(
                "GET", requests.get,
                image_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36"
                },
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 120),
                stream=True,
            )
            if response.status_code == 200:
                try:
                    declared_size = int(response.headers.get("Content-Length", 0))
                except (TypeError, ValueError):
                    declared_size = 0
                if declared_size > OPENAI_IMAGE_MAX_BYTES:
                    return None, "generated image exceeds the 25 MB download limit"
                image_bytes = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    if len(image_bytes) + len(chunk) > OPENAI_IMAGE_MAX_BYTES:
                        return None, "generated image exceeds the 25 MB download limit"
                    image_bytes.extend(chunk)
                if image_bytes:
                    return bytes(image_bytes), ""
                failure_detail = "generated image download was empty"
            else:
                failure_detail = f"HTTP {response.status_code} while downloading image"
        except Exception as e:
            failure_detail = (
                f"error={type(e).__name__}, "
                f"detail={_redact_request_error(e, api_key, image_url)}"
            )
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception as exc:
                    logger.warning(
                        "failed to close generated image download response: "
                        f"error={type(exc).__name__}"
                    )
        if attempt < OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS:
            logger.warning(
                "generated image download failed, retrying the same url: "
                f"attempt={attempt}/{OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS}, "
                f"{failure_detail}"
            )
            time.sleep(OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS)
    raise OpenAIImagePaidResultError(
        "generated image could not be downloaded after retries"
    )


def _parse_openai_image_response(
    response: Any,
    api_key: str,
) -> tuple[bytes | None, str]:
    """
    Parse /images/generations response, extracting url or b64_json image payload.

    Parsing failures represent explicit business rejection (such as content moderation) or malformed responses;
    return error description directly without backoff retries — repeating identical requests produces identical rejections.
    """
    body = _response_json_safely(response)
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        return None, _redact_secret(_openai_image_response_message(body), api_key)

    entry = data[0]
    if not isinstance(entry, dict):
        return None, "invalid image data entry"

    b64_payload = entry.get("b64_json")
    if b64_payload:
        if (
            not isinstance(b64_payload, str)
            or len(b64_payload) > 4 * ((OPENAI_IMAGE_MAX_BYTES + 2) // 3)
        ):
            return None, "generated image exceeds the 25 MB response limit"
        try:
            image_bytes = base64.b64decode(b64_payload)
        except Exception as e:
            return None, f"invalid b64_json payload: {type(e).__name__}"
        if len(image_bytes) > OPENAI_IMAGE_MAX_BYTES:
            return None, "generated image exceeds the 25 MB response limit"
        return image_bytes, ""

    image_url = entry.get("url")
    if isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
        return _openai_image_download_bytes(image_url, api_key)

    return None, "image response has neither url nor b64_json"


def _request_openai_image(endpoint: str, payload: dict) -> tuple[bytes | None, str]:
    """
    Call OpenAI-compatible /images/generations endpoint with backoff retries and key rotation.

    Retries 429/5xx as transient errors with backoff; 401/403 is retried only when multiple keys are configured
    (leveraging get_api_key rotation); remaining 4xx errors are explicit rejections that fail fast,
    allowing upper layers to skip the keyword.

    Billing safety: POST read timeouts and connection resets are treated as "unconfirmed" state — the server may have
    already generated and billed while the response was lost; auto-resubmitting risks duplicate generation and billing,
    so a dedicated exception is raised to abort the local task. Only connection-phase timeouts (ConnectTimeout,
    guaranteed not to have reached the server) confirm no task creation and can be safely retried.

    API Key may be empty: fully local ComfyUI/SD gateways typically need no authentication and omit Authorization headers when blank.
    """
    api_keys = config.app.get("openai_image_api_keys")
    if isinstance(api_keys, (list, tuple)):
        configured_keys = [k for k in api_keys if str(k or "").strip()]
    elif str(api_keys or "").strip():
        configured_keys = [api_keys]
    else:
        configured_keys = []

    failure_detail = "no request attempt was made"
    for attempt in range(1, OPENAI_IMAGE_MAX_ATTEMPTS + 1):
        api_key = get_api_key("openai_image_api_keys") if configured_keys else ""
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        retryable = False
        try:
            response = trace_http_call(
                "POST", requests.post,
                endpoint,
                json=payload,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=OPENAI_IMAGE_REQUEST_TIMEOUT,
                allow_redirects=False,
            )
        except requests.exceptions.ConnectTimeout as e:
            # Connect timeout: request definitely did not reach the server; no task was created, safe to retry.
            failure_detail = (
                f"connect timeout: detail={_redact_request_error(e, api_key)}"
            )
            retryable = True
        except Exception as e:
            # Read timeout/connection reset is "unconfirmed": server may have accepted and billed;
            # auto-resubmitting or proceeding with subsequent keywords risks duplicate billing, so abort local task.
            raise OpenAIImageUnconfirmedError(
                "unconfirmed image request (no retry to avoid double billing): "
                f"{type(e).__name__}, detail={_redact_request_error(e, api_key)}"
            ) from e
        else:
            status = int(getattr(response, "status_code", 200) or 200)
            if 300 <= status < 400:
                response.close()
                raise OpenAIImageUnconfirmedError(
                    "image submission returned a redirect; no redirected request "
                    "or retry was sent because the paid outcome is unconfirmed"
                )
            if status in OPENAI_IMAGE_KEY_ERROR_STATUS_CODES:
                failure_detail = _openai_image_http_failure(response, status, api_key)
                # Only under multi-key configurations can retries rotate to a valid key.
                retryable = len(configured_keys) > 1
            elif status in OPENAI_IMAGE_RETRYABLE_STATUS_CODES:
                failure_detail = _openai_image_http_failure(response, status, api_key)
                retryable = True
            elif status >= 400:
                return None, _openai_image_http_failure(response, status, api_key)
            else:
                image_bytes, parse_error = _parse_openai_image_response(
                    response, api_key
                )
                if image_bytes is not None:
                    return image_bytes, ""
                failure_detail = parse_error

        if retryable and attempt < OPENAI_IMAGE_MAX_ATTEMPTS:
            backoff_seconds = OPENAI_IMAGE_RETRY_BACKOFF_SECONDS[
                min(attempt - 1, len(OPENAI_IMAGE_RETRY_BACKOFF_SECONDS) - 1)
            ]
            logger.warning(
                "openai image request failed, retrying: "
                f"attempt={attempt}/{OPENAI_IMAGE_MAX_ATTEMPTS}, "
                f"next_retry_in={backoff_seconds}s, detail={failure_detail}"
            )
            time.sleep(backoff_seconds)
            continue
        return None, failure_detail

    return None, failure_detail


def _save_openai_image_file(
    image_bytes: bytes,
    save_dir: str,
) -> tuple[str, int, int]:
    """
    Normalize generated results to PNG on disk, returning (path, width, height).

    Standardizing to PNG avoids two failure classes: relay services returning WebP/JPEG without reliable extensions,
    and images with malformed metadata causing MoviePy parsing failures (matching local material sanitization
    by normalizing directly at persistence time).
    """
    if not save_dir:
        save_dir = utils.storage_dir("cache_images", create=True)
    elif not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    image_path = os.path.join(save_dir, f"openai-image-{uuid.uuid4().hex[:12]}.png")
    if len(image_bytes) > OPENAI_IMAGE_MAX_BYTES:
        raise _OpenAIImageDecodeError("generated image exceeds the 25 MB limit")

    # Image decoding failure can be demoted to "skip current keyword", but directory permissions, disk exhaustion,
    # and file write failures must be re-raised; otherwise on-demand generation loops will keep creating paid tasks
    # when local storage is unavailable. Image.open only inspects in-memory bytes so OSError here indicates format
    # recognition failure; image.load OSError corresponds to truncated or corrupt image payloads.
    image = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(image_bytes))
            if image.width * image.height > OPENAI_IMAGE_MAX_PIXELS:
                raise ValueError("generated image exceeds the 50 million pixel limit")
            image.load()
    except (
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        if image is not None:
            image.close()
        raise _OpenAIImageDecodeError(f"{type(exc).__name__}: {exc}") from exc

    with image:
        if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
            image = image.convert("RGB")
        # Exclude save from decode exception guard: write errors indicate environment unavailability, so abort
        # immediately to prevent subsequent keywords from creating unpersistable paid images.
        image.save(image_path, format="PNG")
        width, height = image.size
    return image_path, width, height


def generate_images_openai(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    save_dir: str = "",
) -> List[MaterialInfo]:
    """
    Generate an image for a script keyword via OpenAI-compatible text-to-image API and save locally.

    Maintains the same signature and empty-list failure convention as generate_videos_wavespeed. Images have no
    native duration, so ``duration`` stores the target segment duration (seconds) for on-demand download accounting.
    API actual dimensions may differ from requested sizes, so true dimensions are stored in rendition rather than request parameters.
    """
    aspect = VideoAspect(video_aspect)
    clip_duration = max(int(minimum_duration), 1)
    endpoint, model = _openai_image_endpoint()
    image_size = _openai_image_size(aspect)
    payload = {
        "model": model,
        "prompt": _openai_image_prompt(search_term),
        "n": 1,
        "size": image_size,
    }
    logger.info(
        f"generating image via openai-compatible endpoint: model={model}, "
        f"term={search_term!r}, size={image_size}"
    )
    image_bytes, failure_detail = _request_openai_image(endpoint, payload)
    if image_bytes is None:
        logger.error(
            f"openai image generation failed: term={search_term!r}, "
            f"detail={failure_detail}"
        )
        return []

    try:
        image_path, width, height = _save_openai_image_file(image_bytes, save_dir)
    except _OpenAIImageDecodeError as e:
        # Compatibility layers may return 200 with non-image bodies (e.g. HTML error pages disguised as JSON,
        # or gateway fallback pages). Undecodable images represent a failed generation attempt; return empty list
        # per source convention to skip this keyword rather than crashing the entire task.
        logger.error(
            "openai image response is not a decodable image, skipping term: "
            f"term={search_term!r}, error={type(e).__name__}, detail={e}"
        )
        return []
    item = MaterialInfo()
    item.provider = "openai_image"
    item.url = image_path
    item.duration = clip_duration
    item.source_info = {
        "provider": "openai_image",
        "search_term": search_term,
        "rendition": {
            "id": None,
            "width": width,
            "height": height,
        },
    }
    return [item]


def _render_openai_image_video(
    image_path: str, clip_duration: int, image_motion: ImageMotion = ImageMotion.zoom
) -> str:
    """
    Render a generated image into an mp4 clip, reusing the local material "image -> animated clip" pipeline.

    Returns an empty string on render failure, signaling callers to stop subsequent paid requests.
    """
    try:
        return video.render_image_zoom_video(
            image_path, clip_duration, image_motion=image_motion
        )
    except Exception as e:
        logger.error(
            "failed to render generated image as a video clip: "
            f"image={image_path}, error={type(e).__name__}, detail={e}"
        )
        return ""


def _download_videos_openai_image_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    image_motion: ImageMotion = ImageMotion.zoom,
) -> List[str]:
    """
    Sequentially generate OpenAI-compatible image assets per script segment, stopping immediately once required duration is reached.

    Shares billing safety semantics with WaveSpeed on-demand generation: images are billed per unit; generating all candidates
    upfront pays for unused visuals. Each image is immediately rendered into an mp4 clip with effective duration accumulated
    (capped by segment duration, matching stock workflows); once required voice duration is met, no further paid requests are made.
    Explicit single-image rejections may be skipped; indeterminate results, download failures, or render failures must terminate
    the task to prevent subsequent keyword billing.
    """
    # Direct service calls must also validate before paid requests; cannot discover invalid motion params after generation.
    raw_motion = getattr(image_motion, "value", image_motion)
    if isinstance(raw_motion, str):
        raw_motion = raw_motion.lower().replace("-", "_")
    try:
        image_motion = ImageMotion(raw_motion)
    except (ValueError, TypeError):
        image_motion = ImageMotion.auto if raw_motion == "auto" else ImageMotion.zoom
    if not material_directory:
        # Generated images are billed per task and non-reusable; place in task dir by default for traceability.
        material_directory = utils.task_dir(task_id)

    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0

    # Non-positive audio duration breaks the completion check; return empty immediately to avoid endless
    # per-image billing for uncompletable tasks (aligned with Seedance preflight semantics).
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError):
        required_duration = 0.0
    if required_duration <= 0:
        logger.warning(
            "skip openai image generation because required audio duration is "
            f"not positive: duration={audio_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    for search_term in search_terms:
        try:
            items = generate_images_openai(
                search_term=search_term,
                minimum_duration=max_clip_duration,
                video_aspect=video_aspect,
                save_dir=material_directory,
            )
        except OpenAIImagePaidResultError:
            _persist_material_sources(task_id, material_sources)
            raise
        for item in items:
            video_file = _render_openai_image_video(
                item.url, max_clip_duration, image_motion=image_motion
            )
            if not video_file:
                _persist_material_sources(task_id, material_sources)
                raise OpenAIImagePaidResultError(
                    "generated image could not be rendered locally"
                )
            logger.info(f"image material rendered: {video_file}")
            video_paths.append(video_file)
            try:
                material_sources.append(_material_source_record(item, video_file))
            except Exception as source_error:
                # Consistent with stock sources: source record errors must not treat successfully paid and rendered
                # assets as failures, much less block video synthesis.
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=openai_image, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(max_clip_duration, item.duration)
            # Uses >= like WaveSpeed: generating another when exactly meeting duration incurs extra cost.
            # Inner and outer checks must maintain identical semantics.
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated image materials cover the required duration, stop "
                f"generating more images: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"generated and rendered {len(video_paths)} image materials")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _search_videos_with_cache(
    provider: str,
    search_videos: Callable[..., List[MaterialInfo]],
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """
    Handle 24-hour search cache uniformly for all three online stock sources.

    Caching wraps search APIs only, leaving downstream download and deduplication logic unchanged. Empty remote
    results are never cached because current provider interfaces use empty lists for both "no hits" and "request failure";
    until split into explicit result types, retrying next time is safer than caching a transient outage for a day.
    """
    cache_args = {
        "provider": provider,
        "search_term": search_term,
        "minimum_duration": minimum_duration,
        "video_aspect": video_aspect,
    }

    def load_cache_safely() -> List[MaterialInfo] | None:
        try:
            return material_cache.load_material_search_cache(**cache_args)
        except Exception as exc:
            # Cache is an optional optimization; any cache exception must be treated as a miss without blocking
            # standard remote searches on Pexels, Pixabay, or Coverr.
            logger.warning(
                "material search cache read failed, continue with remote search: "
                f"provider={provider}, error={type(exc).__name__}, detail={exc}"
            )
            return None

    def load_matching_cache() -> tuple[List[MaterialInfo] | None, int]:
        cached_items = load_cache_safely()
        if cached_items is None:
            return None, 0

        filtered_cached_items = _filter_materials_by_aspect(
            cached_items,
            video_aspect,
        )
        ignored_count = len(cached_items) - len(filtered_cached_items)
        if ignored_count:
            # Stale caches might include mismatched orientations. Refresh the full candidate set even if few items remain,
            # otherwise the cache lifetime repeatedly reuses the same depleted pool.
            return None, ignored_count
        return filtered_cached_items, 0

    cached_items, ignored_count = load_matching_cache()
    if cached_items is not None:
        return cached_items
    if ignored_count:
        logger.info(
            "material search cache contains mismatched orientations, "
            f"refresh from provider: provider={provider}, term={search_term!r}, "
            f"ignored={ignored_count}"
        )

    cache_lock = material_cache.get_material_search_cache_lock(**cache_args)
    with cache_lock:
        # Wait for threads with identical search terms to complete before re-reading, preventing concurrent remote requests
        # on initial cache misses and mitigating rate-limiting/anti-abuse triggers.
        cached_items, _ = load_matching_cache()
        if cached_items is not None:
            return cached_items

        items = search_videos(
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
        # Providers normally write the current keyword, but mocks, third-party extensions, or legacy code may omit or corrupt it.
        # Cache reading restores this field from the cache key, so remote results are calibrated at the same entry point
        # to guarantee identical provenance between fresh searches and cache hits.
        for item in items:
            if isinstance(item.source_info, dict):
                item.source_info = dict(item.source_info)
                item.source_info["search_term"] = search_term
        if items:
            try:
                material_cache.save_material_search_cache(
                    **cache_args,
                    items=items,
                )
            except Exception as exc:
                logger.warning(
                    "material search cache write failed, use remote results: "
                    f"provider={provider}, error={type(exc).__name__}, detail={exc}"
                )
        return items


def _search_terms_in_parallel(
    search_terms: List[str],
    search_videos: Callable[[str, int, VideoAspect], List[MaterialInfo]],
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> list[tuple[str, List[MaterialInfo]]]:
    """Search keywords in parallel; failed keywords return empty results without blocking other materials."""
    if not search_terms:
        return []

    workers = min(_get_material_concurrency(), len(search_terms))
    if workers == 1:
        results = []
        for search_term in search_terms:
            try:
                items = search_videos(
                    search_term=search_term,
                    minimum_duration=minimum_duration,
                    video_aspect=video_aspect,
                )
            except Exception as exc:
                logger.error(
                    "failed to search material videos: "
                    f"search_term={search_term!r}, "
                    f"error={type(exc).__name__}"
                )
                items = []
            logger.info(f"found {len(items)} videos for '{search_term}'")
            results.append((search_term, items))
        return results

    futures = {}
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="material-search",
    ) as executor:
        for search_term in search_terms:
            futures[executor.submit(
                logging_utils.bind_log_scope(search_videos),
                search_term,
                minimum_duration,
                video_aspect,
            )] = search_term

        results = []
        for future in futures:
            search_term = futures[future]
            try:
                items = future.result()
            except Exception as exc:
                logger.error(
                    "failed to search material videos: "
                    f"search_term={search_term!r}, "
                    f"error={type(exc).__name__}, detail={exc}"
                )
                items = []
            logger.info(f"found {len(items)} videos for '{search_term}'")
            results.append((search_term, items))
        return results


def _report_material_downloaded(
    position: int,
    total: int,
    item: MaterialInfo,
    saved_video_path: str,
    on_downloaded: Callable[[MaterialInfo], None] | None,
) -> None:
    """
    Record material download completion and notify callers to update progress.

    Always invoked within the downloading thread; per-file logging and progress updates operate independently of concurrency levels.
    """
    description = os.path.basename(saved_video_path)
    try:
        size_mb = os.path.getsize(saved_video_path) / _BYTES_PER_MEGABYTE
        description += f" ({size_mb:.1f} MB)"
    except OSError:
        # File size is auxiliary information; inability to read it does not invalidate the download.
        pass
    logger.info(f"downloaded material {position}/{total}: {description}")
    if on_downloaded is None:
        return
    try:
        on_downloaded(item)
    except Exception as exc:
        # Progress is display information only. Callback failures (e.g. temporary backend unavailability) must never
        # invalidate downloaded materials or abort the task.
        logger.warning(
            "failed to report material download progress: "
            f"error={type(exc).__name__}, detail={exc}"
        )


def _covered_duration_reporter(
    progress_callback: Callable[[float], None] | None,
    required_duration: float,
    max_clip_duration: int,
) -> Callable[[MaterialInfo], None] | None:
    """
    Generate per-material callback: converts covered audio duration into a 0.0 - 1.0 download progress ratio.

    Matches download loop termination criteria — each material contributes at most one segment duration; once required duration
    is covered, progress reaches 1.0. Returns None if no progress callback is provided.
    """
    if progress_callback is None:
        return None
    covered_duration = 0.0

    def report(item: MaterialInfo) -> None:
        nonlocal covered_duration
        covered_duration += min(max_clip_duration, item.duration)
        fraction = 1.0
        if required_duration > 0:
            fraction = min(1.0, covered_duration / required_duration)
        progress_callback(fraction)

    return report


def _download_materials_in_parallel(
    materials: List[tuple[str, MaterialInfo]],
    material_directory: str,
    on_downloaded: Callable[[MaterialInfo], None] | None = None,
) -> list[tuple[str, MaterialInfo, str]]:
    """
    Download a round of materials in parallel, preserving the caller-provided candidate order.

    Logs a message and invokes ``on_downloaded`` upon each successful download for caller progress updates.
    Callbacks and logging both occur in the invoking thread.
    """
    if not materials:
        return []

    total = len(materials)
    workers = min(_get_material_concurrency(), total)
    if workers == 1:
        downloaded = []
        for position, (search_term, item) in enumerate(materials, start=1):
            try:
                saved_video_path = save_video(
                    video_url=item.url,
                    save_dir=material_directory,
                )
            except Exception as exc:
                logger.error(
                    "failed to download material video: "
                    f"provider={item.provider}, "
                    f"error={type(exc).__name__}, "
                    f"detail={_redact_request_error(exc, item.url)}"
                )
                continue
            if saved_video_path:
                downloaded.append((search_term, item, saved_video_path))
                _report_material_downloaded(
                    position, total, item, saved_video_path, on_downloaded
                )
        return downloaded

    futures = {}
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="material-download",
    ) as executor:
        for search_term, item in materials:
            futures[executor.submit(
                logging_utils.bind_log_scope(save_video),
                item.url,
                material_directory,
            )] = (search_term, item)

        downloaded = []
        for position, future in enumerate(futures, start=1):
            search_term, item = futures[future]
            try:
                saved_video_path = future.result()
            except Exception as exc:
                logger.error(
                    "failed to download material video: "
                    f"provider={item.provider}, "
                    f"error={type(exc).__name__}, "
                    f"detail={_redact_request_error(exc, item.url)}"
                )
                continue
            if saved_video_path:
                downloaded.append((search_term, item, saved_video_path))
                _report_material_downloaded(
                    position, total, item, saved_video_path, on_downloaded
                )
        return downloaded


def _select_materials_until_duration(
    materials: List[tuple[str, MaterialInfo]],
    max_clip_duration: int,
    audio_duration: float,
    current_duration: float = 0.0,
) -> List[tuple[str, MaterialInfo]]:
    """
    Select a round of materials in candidate order, preventing concurrent downloads exceeding required audio duration.

    Returns a prefix subset of ``materials``: accumulates from the first candidate, stopping once cumulative duration
    first meets or exceeds required duration (including the boundary item). Callers use this prefix to determine
    which candidates were actually attempted in this round.
    """
    selected = []
    total_duration = current_duration
    for material in materials:
        selected.append(material)
        total_duration += min(max_clip_duration, material[1].duration)
        if total_duration > audio_duration:
            break
    return selected


def download_videos(
    task_id: str,
    search_terms: List[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
    match_script_order: bool = False,
    progress_callback: Callable[[float], None] | None = None,
    image_motion: ImageMotion = ImageMotion.zoom,
) -> List[str]:
    """
    Search and download materials required to cover audio duration, returning local file paths.

    ``progress_callback`` is optional: stock materials (Pexels, Pixabay, Coverr) trigger it with a 0.0-1.0 ratio
    after each file to advance task progress bars. On-demand paid generation sources do not report download progress.
    """
    provider = "pexels"
    remote_search_videos = search_videos_pexels
    if source == "pixabay":
        provider = "pixabay"
        remote_search_videos = search_videos_pixabay
    elif source == "coverr":
        provider = "coverr"
        remote_search_videos = search_videos_coverr

    def search_videos(
        search_term: str,
        minimum_duration: int,
        video_aspect: VideoAspect,
    ) -> List[MaterialInfo]:
        return _search_videos_with_cache(
            provider=provider,
            search_videos=remote_search_videos,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""

    if source == "wavespeed":
        # AI generation is billed per call, so it cannot follow the stock workflow of fetching candidates for all keywords
        # before selecting; doing so incurs costs for unused segments. Generation sources operate on-demand per segment,
        # stopping once target duration is met; they bypass the 24-hour search cache because asset URLs are expiring signed links
        # and cache reuse would return identical generated videos across tasks.
        return _download_videos_wavespeed_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "volcengine_seedance":
        # Like WaveSpeed, the Volcano Ark official API creates async paid tasks. It must generate on-demand segment by segment,
        # purchasing only the assets strictly required for current audio duration.
        return _download_videos_seedance_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "ofox":
        # Same on-demand billing semantics as WaveSpeed/Ark: OFox /v1/videos creates async paid tasks and must generate
        # segment-by-segment, stopping once duration is satisfied; asset URLs are expiring temporary direct links bypassing 24h search cache.
        return _download_videos_ofox_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "muapi":
        # MuAPI model endpoints create async paid tasks. Submit segment-by-segment and stop once audio duration is covered,
        # avoiding sending unused script keywords to the remote service.
        return _download_videos_muapi_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "metaso_minimax":
        # MetaSO MiniMax also bills per remote async task. Its request body resembles Volcano Ark, but task query routes
        # and response schemas differ, so it shares local on-demand semantics without reusing provider clients,
        # preventing protocol variations from leaking into the orchestration layer.
        return _download_videos_metaso_minimax_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "openai_image":
        # Same on-demand billing semantics as WaveSpeed: text-to-image bills per unit, generating segment-by-segment and stopping
        # once duration is satisfied. Results are single-use local image files bypassing 24h search cache to avoid different tasks reusing identical images.
        return _download_videos_openai_image_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            image_motion=image_motion,
        )

    if match_script_order:
        return _download_videos_by_script_order(
            task_id=task_id,
            search_terms=search_terms,
            search_videos=search_videos,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            progress_callback=progress_callback,
        )

    valid_video_items = []
    valid_video_urls = []
    found_duration = 0.0
    for search_term, video_items in _search_terms_in_parallel(
        search_terms=search_terms,
        search_videos=search_videos,
        minimum_duration=max_clip_duration,
        video_aspect=video_aspect,
    ):
        for item in video_items:
            if item.url not in valid_video_urls:
                valid_video_items.append(item)
                valid_video_urls.append(item.url)
                found_duration += item.duration

    logger.info(
        f"found total videos: {len(valid_video_items)}, required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )
    video_paths = []
    material_sources: list[dict[str, Any]] = []

    concat_mode_value = getattr(video_concat_mode, "value", video_concat_mode)
    if concat_mode_value == VideoConcatMode.random.value:
        random.shuffle(valid_video_items)

    total_duration = 0.0
    on_downloaded = _covered_duration_reporter(
        progress_callback, audio_duration, max_clip_duration
    )
    pending_items = list(valid_video_items)
    # Default/random paths are also governed by material concurrency config: pre-select candidates per round using serial
    # stop criteria (sequential accumulation, stopping when audio duration is first reached). On full success, downloaded sets
    # match serial logic exactly; failed candidates are supplemented by subsequent ones in the next round.
    while pending_items and total_duration <= audio_duration:
        batch = []
        projected_duration = total_duration
        for item in pending_items:
            batch.append(item)
            projected_duration += min(max_clip_duration, item.duration)
            if projected_duration > audio_duration:
                break
        for item in batch:
            source_info = item.source_info if isinstance(item.source_info, dict) else {}
            logger.info(
                f"downloading {item.provider} video: "
                f"asset_id={source_info.get('asset_id') or 'unknown'}"
            )
        downloaded_materials = _download_materials_in_parallel(
            materials=[("", item) for item in batch],
            material_directory=material_directory,
            on_downloaded=on_downloaded,
        )
        pending_items = pending_items[len(batch):]
        for _, item, saved_video_path in downloaded_materials:
            try:
                if saved_video_path:
                    logger.info(f"video saved: {saved_video_path}")
                    video_paths.append(saved_video_path)
                    try:
                        material_sources.append(
                            _material_source_record(item, saved_video_path)
                        )
                    except Exception as source_error:
                        # Source record exceptions must not mark successfully downloaded assets as failures, much less
                        # block video synthesis; retain provider and exception type for troubleshooting.
                        logger.warning(
                            "failed to prepare material source record: "
                            f"provider={item.provider}, "
                            f"error={type(source_error).__name__}, detail={source_error}"
                        )
                    seconds = min(max_clip_duration, item.duration)
                    total_duration += seconds
                    if total_duration > audio_duration:
                        logger.info(
                            f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                        )
                        break
            except Exception as e:
                logger.error(
                    "failed to download material video: "
                    f"provider={item.provider}, error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, item.url)}"
                )
    logger.success(f"downloaded {len(video_paths)} videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_wavespeed_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """
    Sequentially generate WaveSpeed materials per script segment, stopping once required duration is met.

    Each keyword corresponds naturally to a script segment, billed upon generation: generating everything upfront pays for
    unused clips. Here each clip is immediately downloaded and accumulated towards effective duration (capped by segment duration,
    matching stock logic); once target audio duration is covered, no further generation requests are sent. Explicitly failed segments
    may be skipped; indeterminate or download-failed paid tasks must retain their IDs and abort the local task.
    """
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("WaveSpeed audio duration must be finite") from exc
    if not math.isfinite(required_duration):
        raise ValueError("WaveSpeed audio duration must be finite")
    if required_duration <= 0:
        logger.warning(
            "skip WaveSpeed paid generation because required audio duration "
            f"is not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("WaveSpeed clip duration must be a positive integer") from exc
    if clip_duration <= 0:
        raise ValueError("WaveSpeed clip duration must be a positive integer")

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = generate_videos_wavespeed(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except WaveSpeedUnconfirmedTaskError as e:
            # Submitted paid task is in indeterminate state: remote may still be running or completed and billed.
            # Continuing to order subsequent keywords causes redundant generation and double billing; halt here and
            # pass prediction id to task layer for failure reporting, allowing users to retrieve outputs.
            logger.error(
                "stop submitting new wavespeed tasks, the last submitted task "
                f"is unconfirmed: prediction_id={e.prediction_id or 'unknown'}, "
                f"detail={e}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "wavespeed"
            )
            if not saved_video_path:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                prediction_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise WaveSpeedDownloadError(
                    "WaveSpeed generated a paid video but the result could not be "
                    f"downloaded: id={prediction_id or 'unknown'}",
                    prediction_id=prediction_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                # Consistent with stock sources: source record errors must not treat successfully paid and downloaded
                # materials as failures, much less block video synthesis.
                logger.warning(
                    "failed to prepare material source record: "
                    f"provider={item.provider}, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            # Uses >= comparison: when cumulative duration reaches required duration it is sufficient; generating further
            # incurs unnecessary cost. Inner and outer checks must maintain identical semantics.
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated materials cover the required duration, stop "
                f"generating more clips: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break
    logger.success(f"generated and downloaded {len(video_paths)} videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_seedance_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Sequentially generate Volcano Ark Seedance materials, stopping paid ordering immediately once audio duration is covered."""
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    # Paid generation loops must validate the two duration metrics controlling iterations first. NaN/Infinity prevents
    # ``total_duration >= audio_duration`` from ever triggering, and non-positive segment duration prevents accumulation growth;
    # both risk generating useless paid tasks for every keyword.
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance audio duration must be a finite number"
        ) from exc
    if not math.isfinite(required_duration):
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance audio duration must be a finite number"
        )
    if required_duration <= 0:
        logger.warning(
            "skip Seedance paid generation because required audio duration is "
            f"not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance clip duration must be a positive integer"
        ) from exc
    if clip_duration <= 0:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance clip duration must be a positive integer"
        )

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = volcengine_seedance.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except volcengine_seedance.VolcEngineSeedanceUnconfirmedTaskError as exc:
            # Remote paid task may still succeed. Cease further ordering immediately and preserve task ID so users
            # can confirm or retrieve results from the Volcano Ark console.
            logger.error(
                "stop submitting new Seedance tasks because the last paid task "
                f"is unconfirmed: task_id={exc.task_id or 'unknown'}, detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except volcengine_seedance.VolcEngineSeedanceError as exc:
            logger.error(f"Seedance generation failed before completion: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "volcengine_seedance"
            )
            if not saved_video_path:
                # Remote task completed and incurred charges; if local download fails, propagate the remote task ID
                # back to the task state so users can retrieve it via Ark console. Raise dedicated error to prevent subsequent keyword orders.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise volcengine_seedance.VolcEngineSeedanceDownloadError(
                    "Seedance generated a paid video but the result could not be "
                    f"downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=volcengine_seedance, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated Seedance materials cover the required duration; stop "
                f"submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(
        f"generated and downloaded {len(video_paths)} Volcano Engine Seedance videos"
    )
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_ofox_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Sequentially generate OFox materials, stopping paid ordering immediately once audio duration is covered."""
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    # Paid generation loops must validate the two duration metrics controlling iterations first. NaN/Infinity prevents
    # ``total_duration >= audio_duration`` from ever triggering, and non-positive segment duration prevents accumulation growth;
    # both risk generating useless paid tasks for every keyword.
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise ofox.OFoxError("OFox audio duration must be a finite number") from exc
    if not math.isfinite(required_duration):
        raise ofox.OFoxError("OFox audio duration must be a finite number")
    if required_duration <= 0:
        logger.warning(
            "skip OFox paid generation because required audio duration is "
            f"not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ofox.OFoxError("OFox clip duration must be a positive integer") from exc
    if clip_duration <= 0:
        raise ofox.OFoxError("OFox clip duration must be a positive integer")

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = ofox.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except ofox.OFoxUnconfirmedTaskError as exc:
            # Remote paid task may still succeed. Cease further ordering immediately and preserve task ID so users
            # can confirm or retrieve results from the OFox console.
            logger.error(
                "stop submitting new OFox tasks because the last paid task "
                f"is unconfirmed: task_id={exc.task_id or 'unknown'}, detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except ofox.OFoxError as exc:
            logger.error(f"OFox generation failed before completion: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        # When a single keyword explicitly fails remotely (e.g. content moderation), return empty list: task is finished
        # with no billing ambiguity; skip this segment and proceed with subsequent keywords.
        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "ofox"
            )
            if not saved_video_path:
                # Remote task completed and incurred charges; if local download fails, propagate the remote task ID
                # back to task state for console recovery. Raise dedicated error to prevent subsequent keyword orders.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise ofox.OFoxDownloadError(
                    "OFox generated a paid video but the result could not be "
                    f"downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=ofox, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated OFox materials cover the required duration; stop "
                f"submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"generated and downloaded {len(video_paths)} OFox videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_muapi_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Generate MuAPI materials until the narration duration is covered."""
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise muapi.MuAPIError(
            "MuAPI audio duration must be a finite number"
        ) from exc
    if not math.isfinite(required_duration):
        raise muapi.MuAPIError("MuAPI audio duration must be a finite number")
    if required_duration <= 0:
        logger.warning(
            "skip MuAPI paid generation because required audio duration is "
            f"not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise muapi.MuAPIError(
            "MuAPI clip duration must be a positive integer"
        ) from exc
    if clip_duration <= 0:
        raise muapi.MuAPIError("MuAPI clip duration must be a positive integer")

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = muapi.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except muapi.MuAPIUnconfirmedTaskError as exc:
            # An ambiguous paid submission must stop the loop: retrying with the
            # next keyword could create a duplicate charge.  The task ID is kept
            # for task-service recovery when the provider returned one.
            logger.error(
                "stop submitting new MuAPI tasks because the last paid task "
                f"is unconfirmed: task_id={exc.task_id or 'unknown'}, "
                f"detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except muapi.MuAPIError as exc:
            logger.error(f"MuAPI generation failed before completion: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "muapi"
            )
            if not saved_video_path:
                # The remote job completed and may already have incurred a
                # charge.  Do not submit a replacement task after download
                # failure; return the remote ID for manual recovery instead.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise muapi.MuAPIDownloadError(
                    "MuAPI generated a paid video but the result could not be "
                    f"downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=muapi, error={type(source_error).__name__}, "
                    f"detail={source_error}"
                )
            try:
                downloaded_duration = _get_downloaded_video_duration(saved_video_path)
            except Exception as duration_error:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise muapi.MuAPIDownloadError(
                    "MuAPI generated a paid video but its downloaded duration "
                    "could not be measured: "
                    f"id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                ) from duration_error
            total_duration += min(clip_duration, downloaded_duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated MuAPI materials cover the required duration; stop "
                f"submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"generated and downloaded {len(video_paths)} MuAPI videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_metaso_minimax_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Sequentially generate MetaSO MiniMax materials, stopping paid ordering immediately once audio duration is covered."""
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    # Remote generation produces at least 4 seconds, but local pipeline clips and accumulates based on user segment duration.
    # Validate loop control parameters beforehand to prevent NaN, Infinity, or non-positive values from keeping termination
    # unreachable and submitting paid tasks for all keywords.
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax audio duration must be a finite number"
        ) from exc
    if not math.isfinite(required_duration):
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax audio duration must be a finite number"
        )
    if required_duration <= 0:
        logger.warning(
            "skip Metaso MiniMax paid generation because required audio duration "
            f"is not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax clip duration must be a positive integer"
        ) from exc
    if clip_duration <= 0:
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax clip duration must be a positive integer"
        )

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = metaso_minimax.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except metaso_minimax.MetasoMiniMaxUnconfirmedTaskError as exc:
            # When request or polling status is unknown, remote tasks may still succeed and be billed. Immediately halt the entire
            # generation loop to prevent further keyword orders, passing task ID to task service for persistence.
            logger.error(
                "stop submitting new Metaso MiniMax tasks because the last paid "
                f"task is unconfirmed: task_id={exc.task_id or 'unknown'}, "
                f"detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except metaso_minimax.MetasoMiniMaxError as exc:
            logger.error(f"Metaso MiniMax generation failed before completion: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "metaso_minimax"
            )
            if not saved_video_path:
                # Generation succeeded and incurred charges; if download fails, do not create replacement tasks.
                # Raise dedicated error carrying remote ID for task status and manual recovery.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise metaso_minimax.MetasoMiniMaxDownloadError(
                    "Metaso MiniMax generated a paid video but the result could "
                    f"not be downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=metaso_minimax, error={type(source_error).__name__}, "
                    f"detail={source_error}"
                )

            # Local final cut only uses the user-selected segment length; even if H3 produces longer footage due to minimum
            # duration constraints, unused duration must not count towards covered duration to avoid starving necessary visuals.
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated Metaso MiniMax materials cover the required duration; "
                f"stop submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"generated and downloaded {len(video_paths)} Metaso MiniMax videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_by_script_order(
    task_id: str,
    search_terms: List[str],
    search_videos,
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    progress_callback: Callable[[float], None] | None = None,
) -> List[str]:
    """
    Download materials following script narrative order.

    Default download logic concatenates candidate materials from all keywords into one large list; if the first keyword
    returns many results, downloading may exhaust that keyword's materials while later script themes miss the timeline.
    Here we group candidates by keyword and download round-robin:
    Round 1 takes the 1st candidate from each keyword, round 2 takes the 2nd candidate, and so forth.
    This keeps material progression aligned with narration flow without rewriting the video synthesis engine.
    """
    logger.info("downloading videos with script-order material matching")
    candidate_groups = []
    valid_video_urls = set()
    found_duration = 0.0

    for search_term, video_items in _search_terms_in_parallel(
        search_terms=search_terms,
        search_videos=search_videos,
        minimum_duration=max_clip_duration,
        video_aspect=video_aspect,
    ):
        term_items = []
        for item in video_items:
            if item.url in valid_video_urls:
                continue
            term_items.append(item)
            valid_video_urls.add(item.url)
            found_duration += item.duration

        if term_items:
            candidate_groups.append((search_term, term_items))

    logger.info(
        f"found total ordered video candidates: {sum(len(items) for _, items in candidate_groups)}, "
        f"required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )

    video_paths = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0
    on_downloaded = _covered_duration_reporter(
        progress_callback, audio_duration, max_clip_duration
    )
    # Advance candidate indices independently per keyword: only candidates actually chosen for download in this round advance;
    # unchosen candidates remain for the next round, preventing them from being skipped by a global round index.
    next_candidate_indices = [0] * len(candidate_groups)
    while candidate_groups and total_duration <= audio_duration:
        round_materials = [
            (
                group_position,
                search_term,
                term_items[next_candidate_indices[group_position]],
            )
            for group_position, (search_term, term_items) in enumerate(
                candidate_groups
            )
            if next_candidate_indices[group_position] < len(term_items)
        ]
        if not round_materials:
            break

        selected_materials = _select_materials_until_duration(
            materials=[
                (search_term, item) for _, search_term, item in round_materials
            ],
            max_clip_duration=max_clip_duration,
            audio_duration=audio_duration,
            current_duration=total_duration,
        )
        # _select_materials_until_duration returns a prefix subset of candidate materials;
        # thus the first len(selected_materials) items represent candidates actually attempted this round,
        # and only the indices of groups corresponding to these candidates are advanced.
        for group_position, _, _ in round_materials[: len(selected_materials)]:
            next_candidate_indices[group_position] += 1

        downloaded_materials = _download_materials_in_parallel(
            materials=selected_materials,
            material_directory=material_directory,
            on_downloaded=on_downloaded,
        )
        for search_term, item, saved_video_path in downloaded_materials:
            try:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                logger.info(
                    f"downloading ordered {item.provider} video for {search_term!r}: "
                    f"asset_id={source_info.get('asset_id') or 'unknown'}"
                )
                if saved_video_path:
                    logger.info(f"video saved: {saved_video_path}")
                    video_paths.append(saved_video_path)
                    try:
                        material_sources.append(
                            _material_source_record(item, saved_video_path)
                        )
                    except Exception as source_error:
                        logger.warning(
                            "failed to prepare ordered material source record: "
                            f"provider={item.provider}, "
                            f"error={type(source_error).__name__}, "
                            f"detail={source_error}"
                        )
                    total_duration += min(max_clip_duration, item.duration)
                    if total_duration > audio_duration:
                        logger.info(
                            f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                        )
                        break
            except Exception as e:
                logger.error(
                    "failed to download ordered material video: "
                    f"provider={item.provider}, error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, item.url)}"
                )

    logger.success(f"downloaded {len(video_paths)} ordered videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


if __name__ == "__main__":
    download_videos(
        "test123", ["Money Exchange Medium"], audio_duration=100, source="pixabay"
    )
