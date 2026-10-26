import itertools
import hashlib
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
import random
import gc
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from contextlib import ExitStack, contextmanager, redirect_stdout
from functools import lru_cache
from typing import Any, Callable, List, Union
from loguru import logger
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

from app.config import config
from app.models import const
from app.models.schema import (
    ImageMotion,
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import bgm as bgm_service
from app.services.utils import video_effects
from app.utils import file_security, logging_utils, utils
from app.utils.subtitle_reader import moviepy_subtitles

class SubClippedVideoClip:
    def __init__(
        self,
        file_path,
        start_time=None,
        end_time=None,
        width=None,
        height=None,
        duration=None,
        source_file_path=None,
    ):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        self.source_file_path = source_file_path or file_path
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
# FFmpeg/AAC combinations in Docker are more prone to audio quality fluctuations under default settings;
# explicitly increase audio bitrate here to avoid noticeable distortion in the final render caused by low defaults.
audio_bitrate = "192k"
fps = 30
# When FFmpeg concatenates/transcodes by frame rate, the final duration may be tens of milliseconds shorter than MoviePy's theoretical duration.
# Leave a small safety margin on video footage to prevent black frames, stuttering, or narration ending without footage due to frame rounding.
_VIDEO_DURATION_SAFETY_MARGIN = 0.1
_MIN_MATERIAL_DIMENSION = 480
# Messaging apps and certain encoders round down video dimensions; for example, WhatsApp compresses 9:16 assets to 478x850,
# two pixels short of 480. Hard-checking 480 causes these assets to be discarded, failing with "no valid materials found".
# A small tolerance allows footage rounded slightly below threshold while still filtering genuine low-res media.
_MIN_DIMENSION_TOLERANCE = 10
# Process segments sequentially by default to avoid sudden spikes in CPU and memory usage after upgrading.
# Users can increase concurrency for acceleration, but each worker starts an independent encoding task.
_CLIP_PROCESSING_CONCURRENCY = 1


def _get_clip_processing_concurrency() -> int:
    try:
        concurrency = int(config.app.get("video_clip_concurrency", _CLIP_PROCESSING_CONCURRENCY))
    except (TypeError, ValueError):
        concurrency = _CLIP_PROCESSING_CONCURRENCY
    return max(1, min(8, concurrency))
_DEFAULT_VIDEO_CODEC = "libx264"
# FFmpeg emits no intermediate progress logs during concatenation, and `subprocess.run` blocks until process exit,
# making lengthy concatenation appear silent in logs. Periodically log heartbeat messages to distinguish active encoding from freezes.
_FFMPEG_CONCAT_HEARTBEAT_SECONDS = 30.0
# MoviePy similarly emits no logs when rendering the final video, which can take several minutes on CPU for a one-minute vertical video.
# Follow the concatenation pattern by logging periodic heartbeat messages.
_STAGE_HEARTBEAT_SECONDS = 30.0
_DEFAULT_FFMPEG_CONCAT_TIMEOUT_SECONDS = 3600
_SUBTITLE_SPRING_DURATION_SECONDS = 0.18
_MIN_SUBTITLE_SPRING_SCALE = 0.05
_MAX_SUBTITLE_SPRING_SCALE = 1.35
_SUPPORTED_VIDEO_CODECS = (
    "libx264",
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "h264_mf",
    "h264_videotoolbox",
)
_runtime_disabled_video_codecs = set()
# MoviePy pipes sRGB frames, and ffmpeg's default RGB→YUV matrix is BT.601 with no color tags,
# while players and YouTube decode untagged HD as BT.709 and shift the colors. Convert with the
# BT.709 matrix and tag the stream; setparams writes the tags the -color_* flags alone do not.
# The pixel format stays the encoder's choice: MoviePy leaves odd-sized frames to libx264.
_BT709_VIDEO_FILTER = (
    "scale=out_color_matrix=bt709:out_range=tv,"
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
)
_BT709_FFMPEG_PARAMS = ["-vf", _BT709_VIDEO_FILTER]


def _get_subtitle_spring_scale(time_seconds: float, duration_seconds: float) -> float:
    """Return the scaling factor for subtitle bounce animation at a given timestamp."""
    if duration_seconds <= 0 or time_seconds >= duration_seconds:
        return 1.0

    progress = max(0.0, min(time_seconds / duration_seconds, 1.0))
    scale = 1.0 - math.exp(-6.0 * progress) * math.cos(2.5 * math.pi * progress)
    return max(
        _MIN_SUBTITLE_SPRING_SCALE,
        min(scale, _MAX_SUBTITLE_SPRING_SCALE),
    )


def _scale_subtitle_frame_on_canvas(frame: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale subtitle frames or transparent masks around the center while preserving canvas dimensions.

    MoviePy stores subtitle color frames and transparency masks separately. The bounce animation must apply
    identical scaling and cropping to both, otherwise the first animated frame will composite transparent areas
    as black text outlines onto the video. A 2D array represents a 0-1 mask; a 3D array represents an RGB/RGBA frame.
    """
    if frame.ndim not in (2, 3):
        raise ValueError("subtitle frame must be a 2D mask or 3D color frame")

    height, width = frame.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    offset = ((width - scaled_width) // 2, (height - scaled_height) // 2)

    if frame.ndim == 2:
        # MoviePy masks use 0-1 floats, whereas Pillow L mode uses 0-255; restore the original type and range
        # after conversion to maintain CompositeVideoClip transparency semantics.
        mask_image = Image.fromarray(
            np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        )
        resized_mask = mask_image.resize(
            (scaled_width, scaled_height),
            Image.Resampling.BILINEAR,
        )
        mask_canvas = Image.new("L", (width, height), 0)
        mask_canvas.paste(resized_mask, offset)
        return (np.asarray(mask_canvas) / 255.0).astype(frame.dtype, copy=False)

    if frame.shape[2] not in (3, 4):
        raise ValueError("subtitle color frame must use RGB or RGBA channels")
    color_image = Image.fromarray(frame)
    resized_color = color_image.resize(
        (scaled_width, scaled_height),
        Image.Resampling.BILINEAR,
    )
    background = (0, 0, 0, 0) if frame.shape[2] == 4 else (0, 0, 0)
    color_canvas = Image.new(color_image.mode, (width, height), background)
    color_canvas.paste(resized_color, offset)
    return np.asarray(color_canvas).astype(frame.dtype, copy=False)


def _apply_subtitle_spring_animation(clip, subtitle_duration: float):
    """Scale subtitle color frame and mask synchronously to prevent black initial frames in bounce animation."""
    animation_duration = min(
        _SUBTITLE_SPRING_DURATION_SECONDS,
        max(0.0, subtitle_duration),
    )
    if animation_duration <= 0:
        return clip

    def transform_frame(get_frame, time_seconds):
        frame = get_frame(time_seconds)
        scale = _get_subtitle_spring_scale(time_seconds, animation_duration)
        if scale == 1.0:
            return frame
        return _scale_subtitle_frame_on_canvas(frame, scale)

    # apply_to=["mask"] is crucial: MoviePy defaults to processing color frames only, which caused legacy code
    # to briefly retain full-sized masks when each subtitle appeared, rendering black text outlines.
    return clip.transform(transform_frame, apply_to=["mask"])


def _get_required_video_duration(audio_duration: float) -> float:
    """
    Return target concatenated duration for video materials.

    Use case: video footage must cover the entire narration audio. If footage only "exactly matches"
    audio duration, FFmpeg frame-rate rounding may render the final video slightly short; hence a lightweight
    safety margin is added. Separated into an independent function for testability and margin tuning.
    """
    return max(0.0, float(audio_duration) + _VIDEO_DURATION_SAFETY_MARGIN)


def is_material_resolution_acceptable(width: int, height: int) -> bool:
    """
    Check whether footage resolution meets the minimum threshold for synthesis.

    Nominal minimum is 480x480, but allows dimensions below by `_MIN_DIMENSION_TOLERANCE` pixels
    to accommodate rounding down by messaging apps/encoders (e.g. WhatsApp's 478x850).
    """
    min_dimension = _MIN_MATERIAL_DIMENSION - _MIN_DIMENSION_TOLERANCE
    return width >= min_dimension and height >= min_dimension


def _prioritize_unique_source_clips(
    subclipped_items: List[SubClippedVideoClip],
    concat_mode: VideoConcatMode,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
) -> List[SubClippedVideoClip]:
    """
    Prioritize showing each source asset once, reducing repeated footage occurrences in the final cut.

    Online stock footage often splits a long video into multiple short slices. Legacy logic in random mode
    shuffled all short slices directly, causing multiple slices from the same source file to appear near the beginning
    and middle, which users perceived as repetition. This function only reorders clips: placing the longest clip from each
    source file first, keeping remaining slices as fallbacks. When total duration is insufficient, subsequent slices still
    fill the remaining audio duration to preserve synthesis success. Prioritizing the longest slice avoids picking fragmentary
    tail clips early when sufficient unique material exists.
    """
    if not subclipped_items:
        return []

    concat_mode_value = getattr(concat_mode, "value", concat_mode)
    if concat_mode_value != VideoConcatMode.random.value:
        if source_usage is None:
            return subclipped_items
        if not source_groups:
            return sorted(
                subclipped_items,
                key=lambda item: source_usage.get(item.source_file_path, 0),
            )
        # Keep keyword rounds in order while rotating candidates within each keyword.
        groups = {}
        for item in subclipped_items:
            key = source_groups.get(item.source_file_path, item.source_file_path)
            groups.setdefault(key, []).append(item)
        for items in groups.values():
            items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
        return [
            item
            for row in itertools.zip_longest(*groups.values())
            for item in row
            if item is not None
        ]

    grouped_items: dict[str, list[SubClippedVideoClip]] = {}
    for item in subclipped_items:
        grouped_items.setdefault(item.source_file_path, []).append(item)

    primary_items = []
    overflow_items = []
    for items in grouped_items.values():
        primary_item = max(items, key=lambda item: item.duration)
        primary_items.append(primary_item)
        overflow_items.extend(item for item in items if item is not primary_item)

    random.shuffle(primary_items)
    random.shuffle(overflow_items)
    if source_usage is not None:
        # Stable sorting retains randomness among equally used sources.
        primary_items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
        overflow_items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
    logger.info(
        "prioritized unique video materials, "
        f"sources: {len(grouped_items)}, "
        f"primary clips: {len(primary_items)}, "
        f"fallback clips: {len(overflow_items)}"
    )
    return primary_items + overflow_items


def get_ffmpeg_binary():
    """
    Maintain backwards compatibility for callers resolving FFmpeg paths via the video service.

    Core resolution logic has moved to `app.utils.utils.get_ffmpeg_binary()`, which video, voice,
    and future pipelines should share with uniform priority; this thin wrapper prevents AttributeError
    in legacy tests or external scripts importing `app.services.video.get_ffmpeg_binary`.
    """
    return utils.get_ffmpeg_binary()


def _get_configured_video_codec() -> str:
    """
    Retrieve user-configured video encoder.

    Targeted at advanced users attempting hardware encoding via NVENC/AMF/QSV/VideoToolbox.
    Restricted to a strict allowlist to avoid arbitrary FFmpeg flags where invalid inputs produce
    uncontrollable output formats or deferred pipeline failures.
    """
    configured_codec = str(
        config.app.get("video_codec", _DEFAULT_VIDEO_CODEC) or _DEFAULT_VIDEO_CODEC
    ).strip()
    if configured_codec not in _SUPPORTED_VIDEO_CODECS:
        logger.warning(
            f"unsupported video codec configured: {configured_codec}, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC
    return configured_codec


@lru_cache(maxsize=16)
def _ffmpeg_encoder_exists(ffmpeg_binary: str, codec: str) -> bool:
    """
    Check if the current FFmpeg build declares support for the specified encoder.

    This only proves FFmpeg was compiled with the encoder, not that local hardware and drivers
    can successfully initialize it. A fallback to libx264 is retained if encoding fails at runtime.
    """
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {str(exc)}"
        )
        return False

    if result.returncode != 0:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {(result.stderr or result.stdout or '').strip()}"
        )
        return False
    return codec in result.stdout


def _get_effective_video_codec(preferred_codec: str | None = None) -> str:
    """
    Return the active video encoder for the current invocation.

    When users select a hardware encoder, verify against the FFmpeg encoder list first; if runtime encoding
    has already failed in this process, fall back immediately to avoid failing repeatedly on every clip.
    """
    selected_codec = preferred_codec or _get_configured_video_codec()
    if selected_codec == _DEFAULT_VIDEO_CODEC:
        return _DEFAULT_VIDEO_CODEC

    if selected_codec in _runtime_disabled_video_codecs:
        logger.warning(
            f"video codec {selected_codec} was disabled after a runtime failure, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if not _ffmpeg_encoder_exists(ffmpeg_binary, selected_codec):
        logger.warning(
            f"ffmpeg encoder {selected_codec} is not available, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    return selected_codec


def _disable_runtime_video_codec(codec: str, reason: str):
    if codec == _DEFAULT_VIDEO_CODEC:
        return
    _runtime_disabled_video_codecs.add(codec)
    logger.warning(
        f"video codec {codec} failed, fallback to {_DEFAULT_VIDEO_CODEC}. "
        f"reason: {reason}"
    )


def _get_temp_audio_dir(output_dir: str) -> str:
    """
    Return the directory to use for MoviePy's temporary audio file.

    On Windows, Windows Defender can lock files written to the task output
    directory while scanning them, causing MoviePy to fail with a
    PermissionError (WinError 32) on the TEMP_MPY_wvf_snd temp file and
    leaving the final MP4 at 0 bytes.  Using the system temp directory
    sidesteps the scan without changing behaviour on other platforms.

    On Linux/macOS/Docker the output directory is returned unchanged so
    existing behaviour is preserved.
    """
    if sys.platform == "win32":
        return tempfile.gettempdir()
    return output_dir


def _fallback_write_videofile(clip, output_file: str, failed_codec: str, reason: str, **kwargs):
    """
    Retry with libx264 after hardware encoding failure, disabling the hardware encoder only upon retry success.

    FFmpeg failures on Windows can stem from GPU/driver incompatibility or generic IO issues such as locked files,
    folder permissions, or antivirus interference. Only when libx264 succeeds can we deduce that the original failure
    was likely hardware encoder-specific, preventing false disables for subsequent tasks.
    """
    clip.write_videofile(output_file, codec=_DEFAULT_VIDEO_CODEC, **kwargs)
    _disable_runtime_video_codec(failed_codec, reason)
    return _DEFAULT_VIDEO_CODEC


def _write_videofile_with_codec_fallback(
    clip, output_file: str, codec: str, atomic_output: bool = False, **kwargs
):
    """
    Write video with the specified encoder, falling back to libx264 automatically on failure.

    Hardware encoder availability depends on FFmpeg compilation, GPU hardware, driver support, and host environment.
    Generation tasks must not fail outright due to hardware encoder issues, so fallback logic is centralized here.
    """
    kwargs.setdefault("ffmpeg_params", _BT709_FFMPEG_PARAMS)
    if atomic_output:
        # Final videos can be downloaded by path while they are being rendered.
        # Keep both failed encodes and in-progress writes away from that path.
        output_dir = os.path.dirname(os.path.abspath(output_file))
        descriptor, temp_output = tempfile.mkstemp(
            prefix=f".{os.path.basename(output_file)}.",
            suffix=os.path.splitext(output_file)[1] or ".mp4",
            dir=output_dir,
        )
        os.close(descriptor)
        os.unlink(temp_output)
        try:
            used_codec = _write_videofile_with_codec_fallback(
                clip, temp_output, codec, **kwargs
            )
            os.replace(temp_output, output_file)
            return used_codec
        finally:
            try:
                os.unlink(temp_output)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    f"failed to remove temporary final video: {temp_output}, "
                    f"error: {exc}"
                )

    effective_codec = _get_effective_video_codec(codec)
    try:
        clip.write_videofile(output_file, codec=effective_codec, **kwargs)
        return effective_codec
    except Exception as exc:
        if effective_codec == _DEFAULT_VIDEO_CODEC:
            raise
        return _fallback_write_videofile(
            clip,
            output_file,
            failed_codec=effective_codec,
            reason=str(exc),
            **kwargs,
        )


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # Concat demuxer wraps paths in single quotes; single quotes within paths must be escaped first.
    return file_path.replace("'", "'\\''")


def _format_ffmpeg_concat_path(file_path: str) -> str:
    """
    Format file paths for the FFmpeg concat demuxer manifest.

    FFmpeg documentation specifies that whitespace and special characters in concat files require escaping;
    backslashes in Windows absolute paths can also be misinterpreted as escape characters. Normalize all paths
    to forward slashes (`C:/Users/...` instead of `C:\\Users\\...`) and escape single quotes for cross-platform compatibility.
    """
    absolute_path = os.path.abspath(file_path)
    # Normalize the native separator only. A backslash can be a literal
    # component of a valid POSIX path and must still identify the same file.
    return _escape_ffmpeg_concat_path(absolute_path.replace(os.sep, "/"))


def _describe_concat_output_progress(output_file: str) -> str:
    """Return a human-readable description of current output file size for concatenation heartbeat logs."""
    try:
        size = os.path.getsize(output_file)
    except OSError:
        # Fall back safely if the output file is not yet created without affecting concatenation.
        return "output size not available"
    return f"output size: {size / (1024 * 1024):.2f} MB"


@contextmanager
def _stage_heartbeat(description: str):
    """
    Log periodic heartbeat messages during lengthy stages that lack intrinsic progress output.

    The heartbeat thread binds context to the calling thread so logs appear in the corresponding WebUI task panel.
    Stops and joins the heartbeat thread on phase completion or exception to avoid emitting "still running" after termination.
    """
    started_at = time.monotonic()
    stop_event = threading.Event()

    def log_heartbeat() -> None:
        while not stop_event.wait(_STAGE_HEARTBEAT_SECONDS):
            logger.info(
                f"{description} still running: "
                f"elapsed={time.monotonic() - started_at:.0f}s"
            )

    reporter = threading.Thread(
        target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
    )
    reporter.start()
    try:
        yield
    finally:
        stop_event.set()
        reporter.join(timeout=1)


def _report_clip_progress(
    progress_callback: Callable[[float], None] | None,
    covered_duration: float,
    required_duration: float,
) -> None:
    """Convert covered final duration into a 0.0-1.0 ratio and notify callers."""
    if progress_callback is None:
        return
    fraction = 1.0
    if required_duration > 0:
        fraction = min(1.0, covered_duration / required_duration)
    try:
        progress_callback(fraction)
    except Exception as exc:
        # Progress is display information only; callback failure (e.g. backend temporarily down) must not invalidate processed clips.
        logger.warning(
            "failed to report clip processing progress: "
            f"error={type(exc).__name__}, detail={exc}"
        )


def _run_concat_with_heartbeat(command: list[str], output_file: str):
    """
    Block waiting for FFmpeg completion while periodically logging heartbeat messages.

    FFmpeg emits no intermediate logs during concatenation, and `subprocess.run` buffers output until process exit,
    causing lengthy merges to look silent. Logging elapsed time and output file size clarifies that encoding is progressing.
    """
    started_at = time.monotonic()
    stop_event = threading.Event()

    def log_heartbeat() -> None:
        while not stop_event.wait(_FFMPEG_CONCAT_HEARTBEAT_SECONDS):
            logger.info(
                "ffmpeg concat still running: "
                f"elapsed={time.monotonic() - started_at:.0f}s, "
                f"{_describe_concat_output_progress(output_file)}"
            )

    reporter = threading.Thread(
        target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
    )
    reporter.start()
    try:
        configured_timeout = config.app.get(
            "ffmpeg_concat_timeout_seconds", _DEFAULT_FFMPEG_CONCAT_TIMEOUT_SECONDS
        )
        try:
            timeout_seconds = float(configured_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("ffmpeg_concat_timeout_seconds must be positive") from exc
        if (
            isinstance(configured_timeout, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("ffmpeg_concat_timeout_seconds must be positive")

        try:
            # subprocess.run kills and waits for FFmpeg on timeout, so a stalled
            # encoder cannot leave an orphaned child or a permanently active task.
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"ffmpeg concat exceeded {timeout_seconds:g} seconds"
            ) from exc
    finally:
        stop_event.set()


def concat_video_clips_with_ffmpeg(
    clip_files: List[str],
    output_file: str,
    threads: int,
    output_dir: str,
    max_duration: float | None = None,
):
    # Separate renders may share a directory. Each FFmpeg process must keep its
    # own manifest until all codec attempts finish, without overwriting or
    # deleting another render's list.
    concat_list_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="ffmpeg-concat-", suffix=".txt",
            dir=output_dir, delete=False,
        ) as fp:
            concat_list_file = fp.name
            for clip_file in clip_files:
                fp.write(f"file '{_format_ffmpeg_concat_path(clip_file)}'\n")
    except Exception:
        if concat_list_file:
            delete_files(concat_list_file)
        raise

    staged_output = None

    def build_command(codec: str) -> list[str]:
        command = [
            utils.get_ffmpeg_binary(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            "-c:v",
            codec,
            "-threads",
            str(threads or 2),
            "-vf",
            _BT709_VIDEO_FILTER,
            "-pix_fmt",
            "yuv420p",
        ]
        if max_duration is not None and max_duration > 0:
            command.extend(["-t", f"{max_duration:.3f}"])
        command.append(staged_output)
        return command

    def run_concat(codec: str):
        command = build_command(codec)
        # Use FFmpeg for a single concatenation and encode pass, avoiding repeated re-encodings from MoviePy segment merges
        # and reducing risk of visual degradation or color shift. Heartbeat logs indicate active execution during the wait.
        result = _run_concat_with_heartbeat(command, staged_output)
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or "ffmpeg concat failed")
        return codec

    try:
        descriptor, staged_output = tempfile.mkstemp(
            prefix=".ffmpeg-concat-",
            suffix=os.path.splitext(output_file)[1] or ".mp4",
            dir=os.path.dirname(os.path.abspath(output_file)),
        )
        os.close(descriptor)
        effective_codec = _get_effective_video_codec()
        try:
            result_codec = run_concat(effective_codec)
        except TimeoutError:
            # A hung encoder is not evidence that another codec will work. Do
            # not spend a second timeout period retrying the same input.
            raise
        except Exception as exc:
            if effective_codec == _DEFAULT_VIDEO_CODEC:
                raise
            result_codec = run_concat(_DEFAULT_VIDEO_CODEC)
            _disable_runtime_video_codec(effective_codec, str(exc))
        # Failed attempts and in-progress output stay private until FFmpeg has
        # finished. Publication errors must not trigger another codec attempt.
        if os.path.getsize(staged_output) == 0:
            raise RuntimeError("ffmpeg concat produced no output")
        os.replace(staged_output, output_file)
        return result_codec
    finally:
        delete_files([concat_list_file, staged_output])


def _sanitize_image_file(image_path: str) -> str:
    # Certain local images open fine in Pillow but crash MoviePy ImageClip due to corrupted EXIF/eXIf metadata.
    # Export a "clean image" copy here with corrupted metadata stripped.
    # Bound the intermediate basename while retaining complete source identity.
    # Different extensions must not overwrite pixels, and valid long filenames
    # must not exceed the filesystem's component limit after adding a suffix.
    source_digest = hashlib.sha256(os.fsencode(os.path.abspath(image_path)))

    temp_path = ""
    try:
        with Image.open(image_path) as image:
            with ImageOps.exif_transpose(image) as upright:
                upright.load()
                # Strip metadata after applying the camera's orientation.
                # Palette transparency belongs to pixels, not removable metadata.
                mode = "RGBA" if "A" in upright.getbands() or "transparency" in upright.info else "RGB"
                if upright.mode == "I;16":
                    # PNG grayscale samples span0..65535; an RGB cast either
                    # clips them to white or wraps when MoviePy encodes them.
                    with upright.point(lambda value: value / 257 + 0.5) as byte_range:
                        cleaned_image = byte_range.convert(mode)
                else:
                    cleaned_image = upright.convert(mode)
                cleaned_image.info.clear()
                source_digest.update(str(cleaned_image.size).encode())
                source_digest.update(cleaned_image.mode.encode())
                source_digest.update(cleaned_image.tobytes())
                sanitized_path = os.path.join(
                    os.path.dirname(image_path),
                    f"{source_digest.hexdigest()}.sanitized.png",
                )
                descriptor, temp_path = tempfile.mkstemp(
                    prefix=".image-sanitize-", suffix=".png",
                    dir=os.path.dirname(os.path.abspath(sanitized_path)),
                )
                os.close(descriptor)
                cleaned_image.save(temp_path)
                cleaned_image.close()
        os.replace(temp_path, sanitized_path)
        temp_path = ""
    finally:
        if temp_path:
            delete_files(temp_path)

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str):
    # MoviePy does not apply camera EXIF orientation while decoding an image.
    # CMYK JPEG channels must become RGB; LA must become RGBA so MoviePy
    # creates a proper opacity mask rather than treating alpha as a color band.
    # Bilevel pixels must become byte-range RGB rather than bool values0/1.
    # Ordinary RGB inputs retain the direct path.
    try:
        with Image.open(image_path) as image:
            orientation = image.getexif().get(274, 1)
            image_mode = image.mode
    except Exception:
        orientation = 1
        image_mode = None
    if orientation in range(2, 9) or image_mode in {"CMYK", "LA", "1", "I;16"}:
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path

    # Attempt opening the original image directly first; if corrupted metadata causes failure, generate a clean copy.
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"failed to open image directly, trying sanitized copy: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path


_moviepy_reader_open_lock = threading.Lock()


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    Open video files quietly, preventing MoviePy 2.1.x from dumping FFmpeg probe outputs directly to stdout.

    Background:
    The dependency version's `FFMPEG_VideoReader` contains explicit `print(self.infos)` and `print(ffmpeg command)`
    calls that output `audio_found: False` on intermediate silent videos. This is merely input asset metadata, not an
    indication of missing audio in the final cut, but it misleads WebUI and terminal users into suspecting failure.

    Implementation:
    1. Redirect stdout solely within the brief window of opening VideoFileClip;
    2. Default `audio=False`, as raw stock audio is discarded and the final track is attached during `generate_video()`;
    3. Re-route any library output to debug logs for troubleshooting if needed.
    """
    captured_stdout = io.StringIO()
    # redirect_stdout changes process-wide state. Overlapping reader opens can
    # restore each other's capture buffers instead of the original stdout.
    # Serialize this short construction window; clip processing stays parallel.
    with _moviepy_reader_open_lock:
        with redirect_stdout(captured_stdout):
            clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "suppressed MoviePy video reader stdout for "
            f"{video_path}, chars: {len(moviepy_stdout)}"
        )

    return clip


def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    # When looping footage, the same temporary clip path appears multiple times in the FFmpeg concat manifest.
    # Merging requires retaining duplicates, but cleanup must unlink each file only once; deduplicate paths in original order
    # to ensure idempotent cleanup and avoid consecutive FileNotFoundError warnings after the first deletion.
    unique_files = dict.fromkeys(file for file in files if file)
    for file in unique_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            # File absence during cleanup is expected (e.g. failure rollback or concurrent cleanup);
            # do not pollute generation logs with benign missing file messages.
            continue
        except OSError as e:
            # File system permission, read-only, or disk issues leave leaked temp files; log warnings
            # with specific paths and errors to assist environment troubleshooting.
            logger.warning(f"failed to delete temporary file {file}: {str(e)}")


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        try:
            resolved_bgm_file = bgm_service.resolve_bgm_file(bgm_file)
        except ValueError as exc:
            # API bgm_file originates from user input; restrict resolution strictly to user BGM or built-in
            # directories, preventing MoviePy from loading arbitrary host files such as configurations or keys.
            logger.warning(
                f"reject unsafe bgm file: {bgm_file}, error: {str(exc)}"
            )
            return ""
        return resolved_bgm_file

    if bgm_type == "random":
        files = bgm_service.list_bgm_files()
        # When background music directory is empty, fall back to "no BGM" to prevent random.choice([]) from raising IndexError.
        if not files:
            logger.warning("no background music files found")
            return ""
        return random.choice(files)

    return ""


def _fit_clip_to_canvas(
    clip,
    *,
    target_width: int,
    target_height: int,
    fit_mode: VideoFitMode | str = VideoFitMode.cover,
):
    """Resize a clip to an exact canvas using cover/crop or contain/letterbox."""
    source_width, source_height = (int(value) for value in clip.size)
    target_width = int(target_width)
    target_height = int(target_height)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError(
            "video dimensions must be positive: "
            f"source={source_width}x{source_height}, "
            f"target={target_width}x{target_height}"
        )

    mode = VideoFitMode(fit_mode)
    if (source_width, source_height) == (target_width, target_height):
        return clip

    # Exact aspect-ratio matches do not need either a crop or a background.
    if source_width * target_height == source_height * target_width:
        return clip.resized(new_size=(target_width, target_height))

    width_scale = target_width / source_width
    height_scale = target_height / source_height

    if mode == VideoFitMode.cover:
        # ceil guarantees the resized clip covers the complete canvas despite
        # floating-point rounding. Any excess is removed symmetrically.
        scale_factor = max(width_scale, height_scale)
        resized_width = max(target_width, math.ceil(source_width * scale_factor))
        resized_height = max(target_height, math.ceil(source_height * scale_factor))
        resized_clip = clip.resized(new_size=(resized_width, resized_height))
        crop_x = max(0, (resized_width - target_width) // 2)
        crop_y = max(0, (resized_height - target_height) // 2)
        return resized_clip.cropped(
            x1=crop_x,
            y1=crop_y,
            width=target_width,
            height=target_height,
        )

    # contain preserves the legacy behavior: show the complete source frame,
    # centered over a black canvas when the aspect ratios differ.
    scale_factor = min(width_scale, height_scale)
    resized_width = max(1, min(target_width, int(source_width * scale_factor)))
    resized_height = max(1, min(target_height, int(source_height * scale_factor)))
    background = ColorClip(
        size=(target_width, target_height), color=(0, 0, 0)
    ).with_duration(clip.duration)
    resized_clip = clip.resized(
        new_size=(resized_width, resized_height)
    ).with_position("center")
    return CompositeVideoClip(
        [background, resized_clip], size=(target_width, target_height)
    ).with_duration(clip.duration)


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    video_fit_mode: VideoFitMode = VideoFitMode.cover,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
    used_video_paths: List[str] | None = None,
    progress_callback: Callable[[float], None] | None = None,
    applied_transitions: List[dict] | None = None,
) -> str:
    # Pydantic leaves this optional default as the string "random" unless
    # explicitly provided. Normalize it once for all downstream enum access.
    video_concat_mode = VideoConcatMode(video_concat_mode or VideoConcatMode.random)
    audio_clip = AudioFileClip(audio_file)
    try:
        # Narration audio duration is only needed here to compute footage concatenation length; audio_clip is not reused.
        # Close immediately after reading to prevent handle leaks across early returns or exception paths.
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"audio duration: {audio_duration} seconds")
    logger.info(f"maximum clip duration: {max_clip_duration} seconds")
    required_video_duration = _get_required_video_duration(audio_duration)
    logger.info(
        f"required video duration: {required_video_duration:.2f} seconds "
        f"(audio duration + {_VIDEO_DURATION_SAFETY_MARGIN:.2f}s safety margin)"
    )

    # Handle missing transition modes from direct API calls gracefully to avoid crashing on .value access.
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    normalized_clip_speed = utils.normalize_clip_speed(clip_speed)
    if normalized_clip_speed != 1.0:
        # Log effective normalized values once to aid parameter boundary diagnostics without spamming hot per-segment loops.
        logger.info(f"clip playback speed: {normalized_clip_speed:.2f}x")
    # max_clip_duration governs final playback duration in the rendered video, not source read length.
    # MoviePy playing 1.5s source at 0.5x produces a 3s clip; playing 6s source at 2x also yields 3s.
    # Source read duration must be derived inversely from playback speed before slicing; reading a fixed 3s before slow-down
    # would cause subsequent segments starting at 3s to skip intermediate footage. This ensures continuous, non-overlapping source timelines.
    source_clip_duration = max_clip_duration * normalized_clip_speed
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    fit_mode = VideoFitMode(video_fit_mode)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    subclipped_items = []
    video_duration = 0
    for video_path in video_paths:
        clip = None
        try:
            clip = _open_video_clip_quietly(video_path)
            clip_duration = float(clip.duration)
            clip_w, clip_h = clip.size
            if (
                not math.isfinite(clip_duration)
                or clip_duration <= 0
                or not all(
                    math.isfinite(float(dimension)) and float(dimension) > 0
                    for dimension in (clip_w, clip_h)
                )
            ):
                raise ValueError("invalid video duration or dimensions")
        except Exception as exc:
            logger.warning(
                f"skipping unreadable video source: path={video_path}, error={exc}"
            )
            continue
        finally:
            close_clip(clip)
        
        start_time = 0

        # Image-derived video clips represent single scenes with custom camera motion;
        # they must not be sliced into multiple fragments to prevent repeated image appearances.
        is_image_video = any(
            motion_tag in os.path.basename(video_path)
            for motion_tag in (
                ".zoom-",
                ".zoom_in-",
                ".zoom_out-",
                ".pan_left-",
                ".pan_right-",
                ".tilt_up-",
                ".tilt_down-",
                ".static-",
            )
        )

        while start_time < clip_duration:
            end_time = min(start_time + source_clip_duration, clip_duration)

            # Retain all valid segments.
            # This preserves videos naturally shorter than max_clip_duration without discarding the trailing remainder of long videos.
            if end_time > start_time:
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                        source_file_path=video_path,
                    )
                )

            start_time = end_time
            if is_image_video or video_concat_mode.value == VideoConcatMode.sequential.value:
                break

    subclipped_items = _prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
        **({"source_usage": source_usage, "source_groups": source_groups}
           if source_usage is not None else {}),
    )
        
    logger.debug(f"total subclipped items: {len(subclipped_items)}")
    
    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    def process_one_clip(indexed_item):
        """Crop, adjust speed, apply transitions to a source segment and export; returns None on failure."""
        index, subclipped_item = indexed_item
        source_clip = None
        clip = None
        clip_file = None
        try:
            logger.debug(
                f"processing clip {index + 1}: {subclipped_item.width}x{subclipped_item.height}, "
                f"source: {os.path.basename(subclipped_item.source_file_path)}"
            )
            source_clip = _open_video_clip_quietly(subclipped_item.file_path)
            clip = source_clip.subclipped(
                subclipped_item.start_time, subclipped_item.end_time
            )
            # Playback speed is an intrinsic clip attribute and should be applied before transitions. This ensures 1s transitions
            # (Fade, Slide) do not scale to 0.5s or 2s; final max duration trimming acts as a guard against floating point errors.
            if normalized_clip_speed != 1.0:
                clip = clip.with_speed_scaled(normalized_clip_speed)
            # Normalize every source clip before transitions are applied. In cover mode
            # the clip fills the canvas and the excess edges are cropped; contain keeps
            # the complete source frame and uses black bars for the unused area.
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                clip_ratio = clip.w / clip.h
                video_ratio = video_width / video_height
                logger.debug(
                    "resizing clip, "
                    f"source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, "
                    f"target: {video_width}x{video_height}, ratio: {video_ratio:.2f}, "
                    f"fit_mode: {fit_mode.value}"
                )
                clip = _fit_clip_to_canvas(
                    clip,
                    target_width=video_width,
                    target_height=video_height,
                    fit_mode=fit_mode,
                )

            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            # Short footage and faster playback can leave less than one second.
            # Keep the chosen transition inside the rendered clip's timeline.
            transition_duration = min(1.0, clip.duration)
            applied_transition_name = "None"
            clean_mode = str(transition_value).lower().replace("-", "_") if transition_value else "none"
            if clean_mode.startswith("videotransitionmode."):
                clean_mode = clean_mode.split(".", 1)[1]

            if clean_mode in ("fadein", "fade_in"):
                clip = video_effects.fadein_transition(clip, transition_duration)
                applied_transition_name = "FadeIn"
            elif clean_mode in ("fadeout", "fade_out"):
                clip = video_effects.fadeout_transition(clip, transition_duration)
                applied_transition_name = "FadeOut"
            elif clean_mode in ("slidein", "slide_in"):
                clip = video_effects.slidein_transition(clip, transition_duration, shuffle_side)
                applied_transition_name = f"SlideIn ({shuffle_side})"
            elif clean_mode in ("slideout", "slide_out"):
                clip = video_effects.slideout_transition(clip, transition_duration, shuffle_side)
                applied_transition_name = f"SlideOut ({shuffle_side})"
            elif clean_mode in ("zoomin", "zoom_in"):
                clip = video_effects.zoomin_transition(clip, 1)
                applied_transition_name = "ZoomIn"
            elif clean_mode in ("zoomout", "zoom_out"):
                clip = video_effects.zoomout_transition(clip, 1)
                applied_transition_name = "ZoomOut"
            elif clean_mode in ("crossdissolve", "cross_dissolve"):
                clip = video_effects.cross_dissolve_transition(clip, transition_duration)
                applied_transition_name = "Cross Dissolve"
            elif clean_mode in ("diptoblack", "dip_to_black"):
                clip = video_effects.dip_to_black_transition(clip, transition_duration)
                applied_transition_name = "Dip to Black"
            elif clean_mode in ("diptowhite", "dip_to_white"):
                clip = video_effects.dip_to_white_transition(clip, min(0.5, transition_duration))
                applied_transition_name = "Dip to White"
            elif clean_mode in ("rippledissolve", "ripple_dissolve"):
                clip = video_effects.ripple_dissolve_transition(clip, transition_duration)
                applied_transition_name = "Ripple Dissolve"
            elif clean_mode in ("irisin", "iris_in"):
                clip = video_effects.iris_in_transition(clip, transition_duration)
                applied_transition_name = "Iris In"
            elif clean_mode in ("irisout", "iris_out"):
                clip = video_effects.iris_out_transition(clip, transition_duration)
                applied_transition_name = "Iris Out"
            elif clean_mode in ("irisbox", "iris_box"):
                clip = video_effects.iris_box_transition(clip, transition_duration)
                applied_transition_name = "Iris Box"
            elif clean_mode in ("clockwipe", "clock_wipe"):
                clip = video_effects.clock_wipe_transition(clip, transition_duration)
                applied_transition_name = "Clock Wipe"
            elif clean_mode in ("wipeleft", "wipe_left"):
                clip = video_effects.linear_wipe_transition(clip, transition_duration, "left")
                applied_transition_name = "Wipe Left"
            elif clean_mode in ("wiperight", "wipe_right"):
                clip = video_effects.linear_wipe_transition(clip, transition_duration, "right")
                applied_transition_name = "Wipe Right"
            elif clean_mode in ("crosszoom", "cross_zoom"):
                clip = video_effects.cross_zoom_transition(clip, min(0.6, transition_duration))
                applied_transition_name = "Cross Zoom"
            elif clean_mode in ("jumpcut", "jump_cut"):
                clip = video_effects.jump_cut_transition(clip)
                applied_transition_name = "Jump Cut"
            elif clean_mode == "shuffle":
                shuffle_pool = [
                    ("Cross Dissolve", lambda c: video_effects.cross_dissolve_transition(c, transition_duration)),
                    ("Jump Cut", lambda c: video_effects.jump_cut_transition(c)),
                    ("Iris In", lambda c: video_effects.iris_in_transition(c, transition_duration)),
                    ("Ripple Dissolve", lambda c: video_effects.ripple_dissolve_transition(c, transition_duration)),
                    ("Clock Wipe", lambda c: video_effects.clock_wipe_transition(c, transition_duration)),
                    ("Dip to White", lambda c: video_effects.dip_to_white_transition(c, min(0.5, transition_duration))),
                    ("Wipe Left", lambda c: video_effects.linear_wipe_transition(c, transition_duration, "left")),
                    ("Wipe Right", lambda c: video_effects.linear_wipe_transition(c, transition_duration, "right")),
                    ("Cross Zoom", lambda c: video_effects.cross_zoom_transition(c, min(0.6, transition_duration))),
                    ("Dip to Black", lambda c: video_effects.dip_to_black_transition(c, transition_duration)),
                    (f"SlideIn ({shuffle_side})", lambda c: video_effects.slidein_transition(c, transition_duration, shuffle_side)),
                    ("ZoomIn", lambda c: video_effects.zoomin_transition(c, 1)),
                ]
                applied_transition_name, chosen_func = random.choice(shuffle_pool)
                clip = chosen_func(clip)
            elif clean_mode in ("auto", "auto_transition"):
                auto_sequence = [
                    ("Cross Dissolve", lambda c: video_effects.cross_dissolve_transition(c, transition_duration)),
                    ("Jump Cut", lambda c: video_effects.jump_cut_transition(c)),
                    ("Iris In", lambda c: video_effects.iris_in_transition(c, transition_duration)),
                    ("Linear Wipe (Left)", lambda c: video_effects.linear_wipe_transition(c, transition_duration, "left")),
                    ("Ripple Dissolve", lambda c: video_effects.ripple_dissolve_transition(c, transition_duration)),
                    (f"SlideIn ({shuffle_side})", lambda c: video_effects.slidein_transition(c, transition_duration, shuffle_side)),
                    ("Cross Zoom", lambda c: video_effects.cross_zoom_transition(c, min(0.6, transition_duration))),
                    ("Clock Wipe", lambda c: video_effects.clock_wipe_transition(c, transition_duration)),
                    ("Dip to White", lambda c: video_effects.dip_to_white_transition(c, min(0.5, transition_duration))),
                    ("Linear Wipe (Right)", lambda c: video_effects.linear_wipe_transition(c, transition_duration, "right")),
                    ("Iris Box", lambda c: video_effects.iris_box_transition(c, transition_duration)),
                    ("Dip to Black", lambda c: video_effects.dip_to_black_transition(c, transition_duration)),
                    ("ZoomIn", lambda c: video_effects.zoomin_transition(c, 1)),
                    ("FadeIn", lambda c: video_effects.fadein_transition(c, transition_duration)),
                ]
                applied_transition_name, chosen_transition = auto_sequence[index % len(auto_sequence)]
                clip = chosen_transition(clip)

            if applied_transitions is not None:
                applied_transitions.append({
                    "clip": index + 1,
                    "transition": applied_transition_name,
                    "file": os.path.basename(subclipped_item.source_file_path),
                })
            logger.info(f"clip {index + 1}: transition applied: {applied_transition_name}")

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)

            # Write each candidate clip to a unique temporary file. Threads must not
            # share the same output path.
            # Distinct combinations can share a task/output directory. Reserve
            # an owned path so their encoders and cleanup never share a clip.
            with tempfile.NamedTemporaryFile(
                dir=output_dir or ".", prefix="temp-clip-", suffix=".mp4", delete=False
            ) as temporary_clip:
                clip_file = temporary_clip.name
            _write_videofile_with_codec_fallback(
                clip,
                clip_file,
                codec=_get_configured_video_codec(),
                logger=None,
                fps=fps,
            )

            clip_duration_saved = clip.duration
            processed_clip = SubClippedVideoClip(
                file_path=clip_file,
                duration=clip_duration_saved,
                width=clip_w,
                height=clip_h,
                source_file_path=subclipped_item.source_file_path,
            )
            clip_file = None
            return processed_clip
        except Exception as exc:
            logger.error(f"failed to process clip: {str(exc)}")
            return None
        finally:
            # The derived clip shares its FFmpeg reader with the source. If
            # subclipping itself failed, close the original source instead.
            close_clip(clip if clip is not None else source_clip)
            # MoviePy may leave a truncated MP4 even when encoding raises. It
            # was never returned, so concat cleanup cannot see it.
            # Close the reader first so Windows can remove the partial file.
            if clip_file:
                delete_files(clip_file)

    # Clips are processed in a thread pool. Per-segment logs provide the sole progress indicators for this phase;
    # propagating thread context allows the WebUI task logger to capture them.
    process_clip_in_task_scope = logging_utils.bind_log_scope(process_one_clip)
    clip_processing_workers = 1
    if len(subclipped_items) >= 2:
        clip_processing_workers = min(_get_clip_processing_concurrency(), len(subclipped_items))
    with ThreadPoolExecutor(
        max_workers=clip_processing_workers,
        thread_name_prefix="clip-process",
    ) as executor:
        next_candidate_index = 0
        while (
            next_candidate_index < len(subclipped_items)
            and video_duration < required_video_duration
        ):
            remaining_duration = required_video_duration - video_duration
            batch = []
            batch_duration = 0.0
            candidate_index = next_candidate_index
            while candidate_index < len(subclipped_items) and batch_duration < remaining_duration:
                subclipped_item = subclipped_items[candidate_index]
                source_duration = subclipped_item.end_time - subclipped_item.start_time
                output_duration = min(
                    max_clip_duration,
                    source_duration / normalized_clip_speed,
                )
                batch.append((candidate_index, subclipped_item))
                batch_duration += output_duration
                candidate_index += 1
            if not batch:
                break
            for processed_clip in executor.map(process_clip_in_task_scope, batch):
                if processed_clip is None:
                    continue
                processed_clips.append(processed_clip)
                video_duration += processed_clip.duration
                # Each 4K segment can take over ten seconds to process; reporting covered duration per clip ensures
                # logs and progress bars show active advancement.
                logger.info(
                    f"processed clip {len(processed_clips)}: "
                    f"{video_duration:.1f} of {required_video_duration:.1f}s covered"
                )
                _report_clip_progress(
                    progress_callback, video_duration, required_video_duration
                )
            next_candidate_index = candidate_index
    
    # loop processed clips until the video duration covers the audio duration and the small safety margin.
    if video_duration < required_video_duration:
        logger.warning(
            f"video duration ({video_duration:.2f}s) is shorter than required duration "
            f"({required_video_duration:.2f}s), looping clips to match audio length."
        )
        base_clips = processed_clips.copy()
        for clip in itertools.cycle(base_clips):
            if video_duration >= required_video_duration:
                break
            processed_clips.append(clip)
            video_duration += clip.duration
        logger.info(
            f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, "
            f"required duration: {required_video_duration:.2f}s, "
            f"looped {len(processed_clips)-len(base_clips)} clips"
        )
     
    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        if video_paths:
            raise RuntimeError("no readable video clips available for merging")
        logger.warning("no clips available for merging")
        return combined_video_path
    
    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"concatenating {len(clip_files)} clips with ffmpeg")
    try:
        concat_video_clips_with_ffmpeg(
            clip_files=clip_files,
            output_file=combined_video_path,
            threads=threads,
            output_dir=output_dir,
            max_duration=audio_duration,
        )
        if used_video_paths is not None:
            # Exclude safety-margin clips that FFmpeg trims entirely from the output.
            elapsed = 0.0
            for clip in processed_clips:
                if elapsed >= audio_duration:
                    break
                used_video_paths.append(clip.source_file_path)
                elapsed += clip.duration
    finally:
        # FFmpeg failures and timeouts must not strand one encoded MP4 per clip.
        # Repeated clips share a path; delete_files already deduplicates them.
        delete_files(clip_files)
            
    logger.info("video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # Subtitle wrapping must occur prior to TextClip instantiation; otherwise MoviePy calculates bounding boxes
    # on unwrapped text. PIL measures rendered widths with the active font/size, constraining lines within visible video width.
    if "\n" in text:
        # Hard breaks in SRT text are separate layout lines. Measuring a token
        # across a newline makes Pillow count both lines as one wide string,
        # then character wrapping can split an otherwise fitting word.
        wrapped_lines = [
            wrap_text(line, max_width, font=font, fontsize=fontsize)
            for line in text.split("\n")
        ]
        return "\n".join(line for line, _ in wrapped_lines), sum(
            height for _, height in wrapped_lines
        )

    font = ImageFont.truetype(font, fontsize)
    max_width = int(max_width)

    # getbbox() returns ink bounding-box height, not font line metrics. For example, text containing only characters
    # without descenders (A, m, n) lacks descent; this discrepancy accumulates over multi-line text and clips the bottom line.
    # Using ascent + descent reflects inherent font geometry independent of glyph composition, matching MoviePy baseline rendering.
    ascent, descent = font.getmetrics()
    line_height = int(ascent + descent)
    if line_height <= 0:
        # Standard TrueType/OpenType fonts will not reach this fallback; diagnostic logs and size fallbacks
        # prevent corrupt or non-standard fonts with zero metrics from rendering invisible subtitles.
        logger.warning(
            "invalid subtitle font metrics, fallback to font size: "
            f"ascent={ascent}, descent={descent}, fontsize={fontsize}"
        )
        line_height = max(1, int(fontsize))

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, line_height
        left, top, right, bottom = font.getbbox(inner_text)
        # bbox remains suitable for measuring line break widths; height must always rely on stable font line metrics.
        return right - left, line_height

    width, height = get_text_size(text)
    if width <= max_width:
        # SRT cues permit author-defined line breaks. Even if lines fit width boundaries without re-wrapping,
        # canvas height must account for existing line count to prevent clipping subsequent lines.
        return text, (text.count("\n") + 1) * line_height

    def attached_text_units(token):
        units = []
        for char in token:
            if units and unicodedata.category(char).startswith("M"):
                units[-1] += char
            else:
                units.append(char)
        return units

    def split_long_token(token):
        # When a single token exceeds line width (common with unspaced Chinese sentences or very long words),
        # fall back to character-level splitting. When candidate overflows, commit the valid current buffer first
        # and start the next line with the current char rather than cramming it into the overflowing line.
        lines = []
        current = ""
        for char in attached_text_units(token):
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            # If the last slice does not fill the line, subsequent tokens should continue appending to it
            # instead of creating extraneous blank lines after long words.
            pieces = split_long_token(word)
            lines.extend(pieces[:-1])
            current = pieces[-1] if pieces else ""

    if current:
        lines.append(current)

    line_start_punctuation = "，。！？；：、,.!?;:)]}）】》」』”’"
    for index in range(1, len(lines)):
        # When splitting unspaced Chinese sentences by character, closing punctuation (period, comma) may be stranded
        # on an orphan line, stretching the background box awkwardly. Move the preceding line's final character down
        # before the punctuation mark so punctuation always clings to text.
        if not lines[index] or lines[index][0] not in line_start_punctuation:
            continue
        if len(lines[index - 1]) <= 1:
            continue

        previous_units = attached_text_units(lines[index - 1])
        if len(previous_units) <= 1:
            continue
        last_unit = previous_units[-1]
        candidate = f"{last_unit}{lines[index]}"
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            lines[index] = candidate
            lines[index - 1] = lines[index - 1][:-len(last_unit)]

    result = "\n".join(line.strip() for line in lines if line.strip()).strip()
    # Height relies on final formatted result. Explicit newlines inside tokens may cause the temporary lines list length
    # to differ from the actual line count rendered by MoviePy.
    height = (result.count("\n") + 1) * line_height
    return result, height


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    # Match the Pillow color syntax accepted by subtitle preflight.
    # The background clip sets opacity separately from the color value.
    if isinstance(color, str):
        try:
            return ImageColor.getcolor(color, "RGB")
        except (ValueError, TypeError):
            pass
    return (0, 0, 0)


def _rounded_subtitle_background_clip(
    width: int,
    height: int,
    color: str,
    alpha: int = 140,
    radius: int = 16,
) -> ImageClip:
    # Rounded subtitle backgrounds are used only when explicitly enabled: render a rounded translucent RGBA plate
    # and pass it to MoviePy as a transparent ImageClip. Preserves default behavior while offering enhanced visual styling.
    rgb = _hex_to_rgb(color)
    safe_alpha = max(0, min(255, int(alpha)))
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, max(0, width - 1), max(0, height - 1)],
        radius=max(0, int(radius)),
        fill=(rgb[0], rgb[1], rgb[2], safe_alpha),
    )
    return ImageClip(np.array(img), transparent=True)


def _get_visible_center_position(
    text_clip: TextClip,
    container_width: int,
    container_height: int,
) -> tuple[int, int]:
    """
    Center TextClip within its background container based on visible ink pixels.

    MoviePy TextClip creates a transparent canvas using font line-height and baseline metrics. Many font glyphs
    do not sit at the geometric center of this canvas; using `with_position("center")` directly centers the entire
    canvas, making text appear vertically off-center. We inspect TextClip's mask to compute offsets strictly from
    the non-transparent ink bounding box, centering visible glyphs inside the background container.
    """
    x = int(round((container_width - text_clip.w) / 2))
    y = int(round((container_height - text_clip.h) / 2))

    try:
        if text_clip.mask is None:
            return x, y

        mask_frame = text_clip.mask.get_frame(0)
        ys, _ = np.where(mask_frame > 0.01)
        if len(ys) == 0:
            return x, y

        visible_top = int(ys.min())
        visible_bottom = int(ys.max())
        visible_height = visible_bottom - visible_top + 1
        y = int(round((container_height - visible_height) / 2 - visible_top))
    except Exception as exc:
        logger.debug(f"failed to center subtitle text by visible mask: {str(exc)}")

    return x, y


def validate_subtitle_colors(params: VideoParams) -> None:
    """Reject colors Pillow cannot render, before generation services run."""
    if not params.subtitle_enabled:
        return
    colors = [("text_fore_color", params.text_fore_color)]
    # Rendering truncates the width to an integer; a zero-width stroke is unused.
    if int(params.stroke_width) > 0:
        colors.append(("stroke_color", params.stroke_color))
    if isinstance(params.text_background_color, str):
        colors.append(("text_background_color", params.text_background_color))
    for field, value in colors:
        if value is None:
            continue
        try:
            ImageColor.getcolor(value, "RGBA")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid subtitle color: {field}") from exc


def subtitle_colors_are_indistinguishable(params: VideoParams) -> bool:
    """Check if subtitle text and background colors match, alerting users to unreadable contrast."""
    if not params.subtitle_enabled or not params.text_background_color:
        return False

    def normalize_color(value):
        if isinstance(value, bool):
            return "#000000" if value else ""
        return str(value or "").strip().lower()

    text_color = normalize_color(params.text_fore_color)
    background_color = normalize_color(params.text_background_color)
    return bool(text_color and text_color == background_color)


@lru_cache(maxsize=64)
def _subtitle_font_supports_sample(font_path: str, sample: str) -> bool:
    """Check if a font covers required glyphs for sample text, caching results."""
    try:
        font = ImageFont.truetype(font_path, 30)
        missing_mask = font.getmask("\U0010ffff")
        missing_signature = (
            missing_mask.size,
            missing_mask.getbbox(),
            bytes(missing_mask),
        )
        for char in sample:
            char_mask = font.getmask(char)
            char_signature = (
                char_mask.size,
                char_mask.getbbox(),
                bytes(char_mask),
            )
            if char_mask.getbbox() is None or char_signature == missing_signature:
                return False
        return True
    except Exception as e:
        # Font probing errors should not halt generation; retain logs for debugging environment issues.
        logger.warning(f"failed to inspect subtitle font glyphs: {font_path}, {e}")
        return True


def subtitle_font_supports_text(font_path: str, text: str) -> bool:
    """Check whether a font can render letters and numbers in text, ignoring whitespace and punctuation."""
    sample = "".join(
        dict.fromkeys(
            char
            for char in str(text or "")
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    )[:64]
    if not sample:
        return True
    return _subtitle_font_supports_sample(font_path, sample)


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
) -> bool:
    """
    Synthesize final video and return whether background music processing succeeded.

    Return value reflects BGM processing status only: returns True when BGM is disabled or successfully mixed;
    returns False if requested BGM failed during loading, effects, or mixing. Video synthesis proceeds with narration
    only upon BGM failure, allowing the orchestrator to decide whether to surface a fallback warning.
    """
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/harry0703/MoneyPrinterTurbo/issues/217
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled:
        if not params.font_name:
            params.font_name = "STHeitiMedium.ttc"
        # Though API routes perform pre-checks, WebUI, CLI, and internal callers enter rendering directly;
        # verify resolved font paths remain within resource/fonts, blocking absolute paths, directory traversal (../),
        # or out-of-tree symlinks before opening via PIL/MoviePy.
        font_path = file_security.resolve_path_within_directory(
            utils.font_dir(), params.font_name
        )
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    def resolve_subtitle_background_color():
        # Backwards compatibility: API `text_background_color` may be boolean or color string.
        # Normalize here to avoid passing raw booleans to TextClip, preventing unpredictable rendering.
        if isinstance(params.text_background_color, bool):
            return "#000000" if params.text_background_color else None
        return params.text_background_color

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        max_width = video_width * 0.9
        bg_color = resolve_subtitle_background_color()
        rounded_bg_enabled = bool(
            getattr(params, "rounded_subtitle_background", False) and bg_color
        )
        has_subtitle_background = bool(bg_color)
        # Rounded backgrounds hug actual text width and use restrained horizontal padding;
        # classic rectangular backgrounds retain generous safety margins to prevent long subtitles from clipping.
        padding_ratio = 0.4 if rounded_bg_enabled else 0.6
        pad_x = int(params.font_size * padding_ratio) if has_subtitle_background else 0
        # Subtitle backgrounds require horizontal inner padding. Deduct padding from available width before wrapping,
        # preventing large fonts or long words from touching container borders. Applied to both rectangular and
        # rounded backgrounds; background-free subtitles retain standard maximum width.
        text_max_width = max(1, int(max_width) - 2 * pad_x)
        wrapped_txt, txt_height = wrap_text(
            phrase,
            max_width=text_max_width,
            font=font_path,
            fontsize=params.font_size,
        )
        interline = int(params.font_size * 0.25)
        line_count = wrapped_txt.count("\n") + 1
        vertical_padding = int(params.font_size * 0.35)
        # Pillow/MoviePy expands stroke borders outward vertically, adding to each line's layout height.
        # Adding a single stroke margin across the whole block causes multi-line text with heavy strokes to accumulate errors.
        # Factoring bilateral stroke padding per line ensures heavy-stroke multi-line subtitles render without clipping.
        stroke_padding = int(params.stroke_width * 2 * line_count)
        text_clip_margin_y = max(
            int(params.font_size * 0.3), int(params.stroke_width * 2)
        )
        # Under `method=label`, MoviePy shrinks text box height automatically, often cutting off bottom halves of the
        # last line on multi-line stroked text. Pass a conservative height accounting for line spacing and vertical padding
        # to ensure the full background and text render intact.
        clip_h = int(
            txt_height
            + vertical_padding
            + (interline * line_count)
            + stroke_padding
        )

        if rounded_bg_enabled:
            # Rounded backgrounds fit text width rather than spanning 90% of screen width. Measure longest line with PIL
            # and add horizontal padding to avoid overly wide backgrounds on short subtitles.
            try:
                font = ImageFont.truetype(font_path, params.font_size)
                text_w = max(
                    int(font.getbbox(line)[2] - font.getbbox(line)[0])
                    for line in wrapped_txt.split("\n")
                )
            except Exception as exc:
                logger.warning(
                    f"failed to measure subtitle text width, fallback to max width: {str(exc)}"
                )
                text_w = int(max_width)

            box_w = max(1, min(int(max_width), text_w + 2 * pad_x))
            radius = max(8, int(params.font_size * 0.4))
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(box_w, None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            clip_h = max(clip_h, text_clip.h)
            bg_clip = _rounded_subtitle_background_clip(
                width=box_w,
                height=clip_h,
                color=bg_color,
                alpha=140,
                radius=radius,
            )
            text_position = _get_visible_center_position(text_clip, box_w, clip_h)
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=(box_w, clip_h),
            )
        elif bg_color:
            size = (
                int(max_width),
                clip_h,
            )
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(int(max_width), None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            size = (size[0], max(size[1], text_clip.h))
            bg_clip = _rounded_subtitle_background_clip(
                width=size[0],
                height=size[1],
                color=bg_color,
                alpha=255,
                radius=0,
            )
            text_position = _get_visible_center_position(text_clip, size[0], size[1])
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=size,
            )
        else:
            size = (
                int(max_width),
                clip_h,
            )
            _clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=size,
                text_align="center",
            )
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        _clip = _clip.with_start(subtitle_item[0][0])
        _clip = _clip.with_end(subtitle_item[0][1])
        _clip = _clip.with_duration(duration)

        # Bounce animation is active only when explicitly chosen; "none" preserves standard subtitle rendering.
        anim_type = getattr(params, "subtitle_animation", "none")
        if anim_type in ("pop_spring", "spring", "pop"):
            _clip = _apply_subtitle_spring_animation(_clip, duration)

        if params.subtitle_position == "bottom":
            _clip = _clip.with_position(("center", video_height * 0.95 - _clip.h))
        elif params.subtitle_position == "top":
            _clip = _clip.with_position(("center", video_height * 0.05))
        elif params.subtitle_position in ("two_thirds_bottom", "two_thirds", "2/3_bottom"):
            # 2/3 from the bottom = 1/3 from the top: y = (video_height - _clip.h) * (1/3)
            y_two_thirds = (video_height - _clip.h) / 3.0
            _clip = _clip.with_position(("center", y_two_thirds))
        elif params.subtitle_position == "custom":
            # Ensure the subtitle is fully within the screen bounds
            margin = 10  # Additional margin, in pixels
            max_y = video_height - _clip.h - margin
            min_y = margin
            custom_y = (video_height - _clip.h) * (params.custom_position / 100)
            custom_y = max(
                min_y, min(custom_y, max_y)
            )  # Constrain the y value within the valid range
            _clip = _clip.with_position(("center", custom_y))
        else:  # center
            _clip = _clip.with_position(("center", "center"))
        return _clip

    # MoviePy CompositeAudioClip.close() does not close child AudioFileClips. Use ExitStack to explicitly hold
    # source audio readers, guaranteeing FFmpeg subprocess release across success, subtitle errors, mix failures,
    # and render crashes, preventing locked file handles on Windows.
    with ExitStack() as clip_stack:
        source_video_clip = clip_stack.enter_context(
            _open_video_clip_quietly(video_path)
        )
        voice_source_clip = clip_stack.enter_context(AudioFileClip(audio_path))
        video_clip = source_video_clip
        audio_clip = voice_source_clip.with_effects(
            [afx.MultiplyVolume(1.0 if params.voice_volume is None else params.voice_volume)]
        )

        def make_textclip(text):
            return TextClip(
                text=text,
                font=font_path,
                font_size=params.font_size,
            )

        if params.subtitle_enabled and subtitle_path and os.path.exists(subtitle_path):
            sub = clip_stack.enter_context(
                SubtitlesClip(
                    subtitles=moviepy_subtitles(subtitle_path),
                    encoding="utf-8",
                    make_textclip=make_textclip,
                )
            )
            text_clips = []
            for item in sub.subtitles:
                clip = create_text_clip(subtitle_item=item)
                text_clips.append(clip)
            video_clip = CompositeVideoClip([video_clip, *text_clips])
            clip_stack.callback(video_clip.close)

        bgm_enabled = bgm_service.should_use_bgm(
            params.bgm_type, params.bgm_volume
        )
        if not bgm_enabled and params.bgm_type:
            # Short-circuit rule shared across all BGM sources: zero or negative volume must not load random, custom,
            # or provider music files, avoiding pointless IO and mixing operations.
            logger.info(
                f"skipping background music because volume is not positive: "
                f"type={params.bgm_type}, volume={params.bgm_volume}"
            )

        # Provider tracks may be supplied directly by orchestrators. None preserves random/custom BGM resolution,
        # while an empty string explicitly disables BGM; all sources remain subject to the global volume check.
        bgm_file = ""
        if bgm_enabled:
            bgm_file = (
                bgm_file_override
                if bgm_file_override is not None
                else get_bgm_file(
                    bgm_type=params.bgm_type,
                    bgm_file=params.bgm_file,
                )
            )
        bgm_mix_succeeded = True
        if bgm_file:
            try:
                bgm_effects = [afx.MultiplyVolume(params.bgm_volume)]
                # Random/custom BGM resolved internally may be shorter than video and requires looping; files passed via
                # orchestrator override indicate pre-adapted duration. Loop decisions key off origin rather than provider name allowlists.
                if bgm_file_override is None:
                    bgm_effects.append(afx.AudioLoop(duration=video_clip.duration))
                # Fade the complete playback timeline, not each source loop.
                bgm_effects.append(afx.AudioFadeOut(3))
                bgm_source_clip = clip_stack.enter_context(AudioFileClip(bgm_file))
                bgm_clip = bgm_source_clip.with_effects(bgm_effects)
                audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
            except Exception:
                bgm_mix_succeeded = False
                # Log full tracebacks and structured context to distinguish audio decoding, MoviePy effects, and
                # CompositeAudioClip errors; file contents and keys are kept out of logs.
                logger.exception(
                    f"failed to mix background music: type={params.bgm_type}, "
                    f"file={bgm_file}"
                )

        final_video_clip = video_clip.with_audio(audio_clip)
        clip_stack.callback(final_video_clip.close)
        # Explicitly preserve input audio sample rate; fall back to MoviePy default 44100Hz if unavailable.
        # Reduces audio fidelity degradation caused by resampling across different environments, especially Docker.
        output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
        with _stage_heartbeat("final video render"):
            _write_videofile_with_codec_fallback(
                final_video_clip,
                output_file=output_file,
                codec=_get_configured_video_codec(),
                atomic_output=True,
                audio_codec=audio_codec,
                audio_fps=output_audio_fps,
                audio_bitrate=audio_bitrate,
                temp_audiofile_path=_get_temp_audio_dir(output_dir),
                threads=params.n_threads or 2,
                logger=None,
                fps=fps,
            )
        return bgm_mix_succeeded


def _normalize_image_motion_str(motion) -> str:
    raw = getattr(motion, "value", motion)
    if isinstance(raw, str):
        clean = raw.lower().replace("-", "_")
        if clean.startswith("imagemotion."):
            clean = clean.split(".", 1)[1]
        if clean in (
            "static",
            "zoom",
            "zoom_in",
            "zoom_out",
            "pan_left",
            "pan_right",
            "tilt_up",
            "tilt_down",
            "auto",
        ):
            return clean
    return "zoom"


def render_image_zoom_video(
    image_path: str, clip_duration: int = 5, image_motion: Any = ImageMotion.zoom
) -> str:
    """
    Render a single image into a static or slowly zooming mp4 clip, returning output path.

    Shared by local material preprocessing and OpenAI-compatible image generation: defaults to ~3% zoom per second,
    while static mode fixes playback duration without motion. Independent of segment transitions. Render exceptions
    are handled by callers per source-specific failure protocols.
    """
    motion_str = _normalize_image_motion_str(image_motion)
    clip, _ = _open_image_clip_with_fallback(image_path)
    # Downscale overly large high-res images (e.g. 4K/8K) to 1080p canvas bounds once before
    # per-frame resize transforms, preventing 30+ second CPU encoding stalls per image.
    max_dim = 1920
    if clip.w > max_dim or clip.h > max_dim:
        scale_down = min(max_dim / clip.w, max_dim / clip.h)
        clip = clip.resized(scale_down)
    clip = clip.with_duration(clip_duration).with_position("center")
    temp_path = ""
    try:
        effective_motion = motion_str
        if motion_str == "auto":
            w, h = clip.size
            ratio = w / h if h > 0 else 1.0
            image_hash = int(hashlib.md5(image_path.encode()).hexdigest(), 16)
            if ratio >= 1.2:
                effective_motion = "pan_right" if (image_hash % 2 == 0) else "pan_left"
            elif ratio <= 0.8:
                effective_motion = "tilt_down" if (image_hash % 2 == 0) else "tilt_up"
            else:
                effective_motion = "zoom_in" if (image_hash % 2 == 0) else "zoom_out"

        composite_size = None
        if effective_motion in ("zoom", "zoom_in"):
            image_clip = clip.resized(
                lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration)
            )
        elif effective_motion == "zoom_out":
            start_scale = 1 + (clip_duration * 0.03)
            image_clip = clip.resized(
                lambda t: max(1.0, start_scale - (clip_duration * 0.03) * (t / clip.duration))
            )
        elif effective_motion == "pan_left":
            scale = 1.15
            scaled = clip.resized(scale)
            dx = scaled.w - clip.w
            image_clip = scaled.with_position(
                lambda t: (-dx + dx * (t / clip.duration), "center")
            )
            composite_size = (clip.w, clip.h)
        elif effective_motion == "pan_right":
            scale = 1.15
            scaled = clip.resized(scale)
            dx = scaled.w - clip.w
            image_clip = scaled.with_position(
                lambda t: (-dx * (t / clip.duration), "center")
            )
            composite_size = (clip.w, clip.h)
        elif effective_motion == "tilt_up":
            scale = 1.15
            scaled = clip.resized(scale)
            dy = scaled.h - clip.h
            image_clip = scaled.with_position(
                lambda t: ("center", -dy + dy * (t / clip.duration))
            )
            composite_size = (clip.w, clip.h)
        elif effective_motion == "tilt_down":
            scale = 1.15
            scaled = clip.resized(scale)
            dy = scaled.h - clip.h
            image_clip = scaled.with_position(
                lambda t: ("center", -dy * (t / clip.duration))
            )
            composite_size = (clip.w, clip.h)
        else:
            image_clip = clip

        final_clip = (
            CompositeVideoClip([image_clip], size=composite_size, bg_color=(0, 0, 0))
            if composite_size
            else CompositeVideoClip([image_clip], bg_color=(0, 0, 0))
        )
        try:
            # The duration changes the rendered content, so it must be part of
            # the output identity. Different tasks may render the same image
            # concurrently; only publish a complete MP4 after MoviePy closes it.
            source_digest = hashlib.sha256(os.fsencode(os.path.abspath(image_path)))
            source_frame = clip.get_frame(0)
            source_digest.update(str(source_frame.shape).encode())
            source_digest.update(source_frame.tobytes())
            if clip.mask is not None:
                source_digest.update(clip.mask.get_frame(0).tobytes())
            source_identity = source_digest.hexdigest()
            duration_identity = hashlib.sha256(str(clip_duration).encode()).hexdigest()[:16]
            video_file = os.path.join(
                os.path.dirname(image_path),
                f"{source_identity}.{effective_motion}-{duration_identity}.mp4",
            )
            descriptor, temp_path = tempfile.mkstemp(
                prefix=".image-zoom-",
                suffix=".mp4",
                dir=os.path.dirname(os.path.abspath(video_file)),
            )
            os.close(descriptor)
            final_clip.write_videofile(
                temp_path, fps=30, logger=None, ffmpeg_params=_BT709_FFMPEG_PARAMS
            )
        finally:
            close_clip(final_clip)
        os.replace(temp_path, video_file)
        temp_path = ""
        return video_file
    finally:
        close_clip(clip)
        if temp_path:
            delete_files(temp_path)


def preprocess_video(
    materials: List[MaterialInfo], clip_duration=4, image_motion: ImageMotion = ImageMotion.zoom
):
    # WebUI may pass an empty material list on re-generation; return empty result immediately to avoid NoneType exceptions.
    if not materials:
        return []

    # Return only verified materials, preventing low-resolution images from progressing to synthesis.
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    image_idx = 0
    for material in materials:
        if not material.url:
            continue

        try:
            material_source_path = file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError as exc:
            # Local video_source paths come from API inputs and must remain confined to the dedicated materials folder.
            # Filenames and legacy absolute paths within the folder are allowed, but directory traversal or host access
            # is blocked to prevent arbitrary file reading or MoviePy probing of sensitive files.
            logger.warning(
                f"skip unsafe local material: {material.url}, "
                f"local_videos_dir: {local_videos_dir}, error: {str(exc)}"
            )
            continue

        ext = utils.parse_extension(material_source_path)
        is_image = ext in const.FILE_TYPE_IMAGES
        try:
            # Open image assets directly as images, avoiding misdetection through VideoFileClip and fragile fallback branches.
            if is_image:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # Fall back to image mode on non-standard extensions or probe failure, maintaining compatibility for legacy image paths.
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
                # The successful decoder determines the material kind, even
                # when the uploaded filename has a video or unknown suffix.
                is_image = True
            except Exception as exc:
                logger.warning(
                    f"skip unreadable local material: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if not is_material_resolution_acceptable(width, height):
                logger.warning(
                    f"low resolution material: {width}x{height}, minimum "
                    f"{_MIN_MATERIAL_DIMENSION}x{_MIN_MATERIAL_DIMENSION} required "
                    f"(tolerance {_MIN_DIMENSION_TOLERANCE}px)"
                )
                # Close handles immediately upon detecting low-resolution assets and omit them from downstream pipelines.
                close_clip(clip)
                continue

            if is_image:
                logger.info(f"processing image: {material_source_path}")
                # Release probe handles opened during dimension inspection before rendering exported image clips.
                close_clip(clip)
                motion_str = _normalize_image_motion_str(image_motion)
                assigned_motion = motion_str
                if motion_str == "auto":
                    ratio = width / height if height > 0 else 1.0
                    if ratio >= 1.25:
                        assigned_motion = (
                            "pan_right"
                            if (image_idx % 2 == 0)
                            else "pan_left"
                        )
                    elif ratio <= 0.8:
                        assigned_motion = (
                            "tilt_down"
                            if (image_idx % 2 == 0)
                            else "tilt_up"
                        )
                    else:
                        auto_sequence = [
                            "zoom_in",
                            "zoom_out",
                            "pan_right",
                            "pan_left",
                        ]
                        assigned_motion = auto_sequence[
                            image_idx % len(auto_sequence)
                        ]
                    image_idx += 1

                logger.info(
                    f"image {image_idx}: applying motion '{assigned_motion}' "
                    f"to {os.path.basename(material_source_path)}"
                )
                video_file = render_image_zoom_video(
                    material_source_path, clip_duration, image_motion=assigned_motion
                )
                material.url = video_file
                material.source_info = material.source_info or {}
                material.source_info["motion"] = assigned_motion
                logger.success(f"image processed: {video_file}")
            else:
                # Standard video footage only requires dimension inspection; release file handles immediately once checked.
                close_clip(clip)
                # Update url to the resolved absolute path so that downstream
                # stages (combine_videos) can open the file without re-resolving.
                material.url = material_source_path
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials
