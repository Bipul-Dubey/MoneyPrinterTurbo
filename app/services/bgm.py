import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from loguru import logger

from app.utils import file_security, utils


# Streamlit allows larger upload sizes by default, but background music is typically only a few megabytes.
# Setting an explicit server-side limit prevents APIs or the WebUI from writing oversized files to disk.
MAX_BGM_UPLOAD_BYTES = 30 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_INTERNAL_UPLOAD_PREFIX = ".bgm-upload-"
_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)}
    # Win32 recognizes Latin-1 superscript digits ¹, ², ³ as device numbers as well; handle them
    # identically to standard digit reserved names to prevent COM¹.mp3 bypasses. Matches webui/Main.py.
    | {f"{prefix}{number}" for prefix in ("COM", "LPT") for number in ("¹", "²", "³")}
)
# Filenames enter logs, API responses, and the WebUI verbatim. Therefore, in addition to path
# separators and Win32 reserved names, all control characters and characters that alter visual
# presentation must be rejected. Previously only ord < 32 (C0 control characters) were blocked,
# missing the other half of the same class of vulnerabilities:
# * C1 control characters U+007F-U+009F: U+0085 is rendered as a newline in some log viewers,
#   allowing a filename to forge extra, independent-looking log lines.
# * Bidirectional control characters U+200E/U+200F/U+202A-U+202E/U+2066-U+2069: characters after
#   U+202E are rendered in reverse, making "photo\u202egnp.mp3" look like a different extension.
# * U+2028/U+2029 (Unicode line/paragraph separators) also create visual line breaks in logs.
# None of these characters can be typed in standard filename dialogs, so normal uploads are unaffected.
# Aligns with app/controllers/base.py's normalize_task_id.
_UNSAFE_FILENAME_CHARACTERS = frozenset(
    chr(code)
    for code in (
        *range(0x00, 0x20),
        *range(0x7F, 0xA0),
        0x200E,
        0x200F,
        0x2028,
        0x2029,
        *range(0x202A, 0x202F),
        *range(0x2066, 0x206A),
    )
)
# MoviePy ultimately decodes BGM via FFmpeg, so restricting to MP3 is unnecessary. Only
# mainstream, semantically unambiguous audio extensions are permitted to avoid uploading video containers.
# The tuple also serves as the single source of truth for WebUI upload components.
SUPPORTED_BGM_EXTENSIONS = (
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
)


class BgmUploadError(ValueError):
    """Indicates that an uploaded file does not satisfy security or format requirements for background music."""


class BgmServiceError(RuntimeError):
    """Indicates server-side execution failures such as unavailable FFmpeg or filesystem errors."""


def should_use_bgm(bgm_type: str | None, bgm_volume: float | None) -> bool:
    """
    Unified check determining whether the current task requires background music processing.

    This rule is source-agnostic: if no source is selected, volume is invalid, or volume <= 0,
    random, custom, Sonilo, and future providers skip parsing, external generation, and final mixing.
    Centralizing this logic avoids duplicating 0-volume checks across providers.
    """
    if not str(bgm_type or "").strip():
        return False
    try:
        normalized_volume = float(bgm_volume or 0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(normalized_volume) and normalized_volume > 0


def uploaded_bgm_dir(create: bool = True) -> str:
    """
    Return the persistent directory for user-uploaded background music.

    Built-in tracks are code assets residing in resource/songs; user uploads are runtime data
    stored under mounted storage to survive container restarts and prevent Git pollution.
    """
    return utils.storage_dir("bgm", create=create)


def _remove_staged_file(file_path: str) -> None:
    """Best-effort cleanup of temporary upload files without masking original exceptions."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        # Temporary files use reserved prefixes and never enter BGM listings. Cleanup failures must
        # not overwrite more specific original exceptions, while logging paths and OS errors for diagnostics.
        logger.warning(
            f"failed to remove staged background music: path={file_path}, "
            f"error={str(exc)}"
        )


def sanitize_upload_filename(filename: str) -> str:
    """Extract a cross-platform displayable audio filename, rejecting invalid names and unsupported extensions."""
    safe_name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if (
        not safe_name
        or safe_name in {".", ".."}
        or len(safe_name) > 255
        or any(character in _UNSAFE_FILENAME_CHARACTERS for character in safe_name)
        or any(character in _WINDOWS_INVALID_FILENAME_CHARS for character in safe_name)
        or safe_name.lower().startswith(_INTERNAL_UPLOAD_PREFIX)
    ):
        raise BgmUploadError("invalid background music filename")

    # Windows treats the segment before the extension as a device name, meaning CON.mp3 or LPT1.wav
    # cannot be created as normal files. Rejecting them upfront ensures consistent cross-platform API behavior.
    windows_basename = safe_name.split(".", 1)[0].rstrip(" .").upper()
    if windows_basename in _WINDOWS_RESERVED_FILENAMES:
        raise BgmUploadError("invalid background music filename")
    if Path(safe_name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
        supported_formats = ", ".join(
            extension.removeprefix(".").upper()
            for extension in SUPPORTED_BGM_EXTENSIONS
        )
        raise BgmUploadError(
            f"unsupported background music format; supported formats: {supported_formats}"
        )
    return safe_name


def _validate_audio(file_path: str, timeout_seconds: int = 30) -> None:
    """
    Verify that the file contains a fully decodable audio stream using only configured FFmpeg.

    imageio-ffmpeg provides portable FFmpeg which does not guarantee FFprobe existence;
    avoid adding a separate binary dependency. `-map 0:a:0` fails if no audio stream exists,
    and `-xerror` promotes decoding errors to failures. Full decoding catches encrypted files or
    random bytes matching audio frame headers. Files may have album art; only the first audio stream is checked.
    """
    try:
        decoded = subprocess.run(
            [
                utils.get_ffmpeg_binary(),
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                # Uploaded files must be self-contained media, not playlists
                # that cause validation/rendering to read adjacent files or URLs.
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                "mp3,mov,aac,wav,flac,ogg,asf",
                "-i",
                file_path,
                "-map",
                "0:a:0",
                # Normalize timestamps from decoded samples. A header-only
                # WAV can exit successfully without producing any audio.
                "-af",
                "asetpts=N/SR/TB",
                "-progress",
                "pipe:1",
                "-nostats",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BgmServiceError("FFmpeg background music validation timed out") from exc
    except OSError as exc:
        raise BgmServiceError("failed to run FFmpeg for background music validation") from exc
    if decoded.returncode != 0:
        raise BgmUploadError("uploaded file must contain a decodable audio stream")
    decoded_audio = False
    for line in decoded.stdout.splitlines():
        key, separator, value = line.partition(b"=")
        if separator and key == b"out_time_us":
            try:
                decoded_audio = decoded_audio or int(value) > 0
            except ValueError:
                continue
    if not decoded_audio:
        raise BgmUploadError("uploaded file must contain nonempty decodable audio")


def validate_audio_file(file_path: str, timeout_seconds: int = 120) -> None:
    """
    Validate that the on-disk audio file can be completely decoded by FFmpeg.

    Upload preflights typically need 30s; Sonilo music can run up to 6 minutes, so an adjustable
    timeout is provided. Relies on FFmpeg only without requiring FFprobe.
    """
    if not os.path.isfile(file_path) or os.path.getsize(file_path) <= 0:
        raise BgmUploadError("background music file is empty or missing")
    _validate_audio(file_path, timeout_seconds=timeout_seconds)


def _stage_bgm_upload(filename: str, source: BinaryIO) -> tuple[str, str, int]:
    """
    Write upload stream to sibling temporary file, returning safe name, temp path, and bytes.

    Preflight and final persistence share identical chunking, size limits, and filename rules
    to prevent status divergence between UI and server. Temp files are deleted or atomically replaced by callers.
    """
    safe_name = sanitize_upload_filename(filename)
    try:
        target_dir = uploaded_bgm_dir(create=True)
    except OSError as exc:
        raise BgmServiceError("failed to prepare background music storage") from exc
    temp_path = ""
    total_bytes = 0

    try:
        try:
            source.seek(0)
        except (AttributeError, OSError) as exc:
            raise BgmUploadError("background music upload is not seekable") from exc

        # Retain original extension so FFmpeg chooses the right demuxer for raw AAC;
        # temp file resides in destination directory ensuring os.replace is atomic.
        descriptor, temp_path = tempfile.mkstemp(
            prefix=_INTERNAL_UPLOAD_PREFIX,
            suffix=Path(safe_name).suffix.lower(),
            dir=target_dir,
        )
        with os.fdopen(descriptor, "wb") as output:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise BgmUploadError("background music upload must be binary")
                total_bytes += len(chunk)
                if total_bytes > MAX_BGM_UPLOAD_BYTES:
                    raise BgmUploadError("background music file exceeds the 30 MB limit")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if total_bytes == 0:
            raise BgmUploadError("background music file is empty")
        return safe_name, temp_path, total_bytes
    except Exception as exc:
        _remove_staged_file(temp_path)
        if isinstance(exc, BgmUploadError):
            raise
        if isinstance(exc, OSError):
            raise BgmServiceError("failed to stage background music upload") from exc
        raise
    finally:
        # Streamlit uses the same UploadedFile for preview; seek(0) ensures playback
        # or subsequent saves do not read empty streams.
        try:
            source.seek(0)
        except (AttributeError, OSError):
            pass


def validate_bgm_upload(filename: str, source: BinaryIO) -> str:
    """Fully validate uploaded audio without persisting, used by WebUI preflight before marking ready."""
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    try:
        _validate_audio(temp_path)
        logger.debug(
            f"background music upload validated: name={safe_name}, "
            f"size={total_bytes} bytes"
        )
        return safe_name
    finally:
        _remove_staged_file(temp_path)


def save_bgm_upload(filename: str, source: BinaryIO) -> str:
    """
    Save background music with chunking, size bounds, and atomic replacement.

    Supports both FastAPI UploadFile and Streamlit UploadedFile. Writes to sibling temp file first
    and validates before atomic os.replace, ensuring unique UUID keys so running tasks reference immutable files.
    """
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    stored_name = f"{uuid4().hex}{Path(safe_name).suffix.lower()}"
    target_path = os.path.join(os.path.dirname(temp_path), stored_name)

    try:
        _validate_audio(temp_path)
        try:
            os.replace(temp_path, target_path)
        except OSError as exc:
            raise BgmServiceError("failed to persist background music upload") from exc
        temp_path = ""
        logger.info(
            f"background music uploaded: original_name={safe_name}, "
            f"stored_name={stored_name}, size={total_bytes} bytes"
        )
        return stored_name
    finally:
        _remove_staged_file(temp_path)


def _list_bgm_files(directories: tuple[str, ...]) -> list[str]:
    """Enumerate safe and supported background music files by directory precedence."""
    files_by_name: dict[str, str] = {}
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory), key=str.lower):
            # Temp files created during upload carry valid audio extensions but are not yet verified;
            # exclude them from candidate random BGM lists.
            if name.startswith(_INTERNAL_UPLOAD_PREFIX):
                continue
            if Path(name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
                continue
            file_path = os.path.join(directory, name)
            try:
                # Validate realpath on enumeration results to prevent symlinks from referencing outside files.
                resolved_path = file_security.resolve_path_within_directory(
                    directory, file_path
                )
            except ValueError as exc:
                logger.warning(
                    f"skip unsafe background music file: name={name}, error={str(exc)}"
                )
                continue
            files_by_name[name] = resolved_path
    return [files_by_name[name] for name in sorted(files_by_name, key=str.lower)]


def list_builtin_bgm_files() -> list[str]:
    """
    List built-in background music distributed with the project.

    WebUI preset songs and preset import/export use this list to ensure portability across machines.
    """
    return _list_bgm_files((utils.song_dir(),))


def list_bgm_files() -> list[str]:
    """List user-uploaded and built-in music, prioritizing uploaded tracks on name collision."""
    return _list_bgm_files((utils.song_dir(), uploaded_bgm_dir(create=True)))


def resolve_builtin_bgm_file(unsafe_path: str) -> str:
    """Resolve built-in music by filename, rejecting path traversals, unknown files, and user uploads."""
    if not unsafe_path:
        raise ValueError("background music filename is required")

    filename = str(unsafe_path)
    if filename != os.path.basename(filename):
        raise ValueError("preset background music must use a filename")

    files_by_name = {
        os.path.basename(file_path): file_path for file_path in list_builtin_bgm_files()
    }
    if filename not in files_by_name:
        raise ValueError("preset background music is not available")
    return files_by_name[filename]


def resolve_bgm_file(unsafe_path: str) -> str:
    """
    Resolve BGM across user upload and built-in directories, rejecting paths outside whitelists
    and unfinished temporary upload files.

    Hits user directory first, while preserving legacy usages like `output000.mp3` or absolute paths.
    New uploads use UUIDs to prevent naming collisions with built-in tracks.
    """
    if (
        not unsafe_path
        or Path(unsafe_path).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS
    ):
        raise ValueError("unsupported background music path")
    if os.path.basename(str(unsafe_path)).lower().startswith(_INTERNAL_UPLOAD_PREFIX):
        # Intermediate ``.bgm-upload-`` files carry audio extensions but are not verified.
        # Exclude them from resolution to prevent targeting incomplete files.
        # User-supplied names are compared case-insensitively, matching Windows/macOS filesystem conventions.
        raise ValueError("background music upload staging files are not selectable")

    candidates = [unsafe_path]
    if not os.path.isabs(unsafe_path):
        candidates.append(os.path.join(utils.root_dir(), unsafe_path))

    last_error = ValueError("background music file does not exist")
    for directory in (uploaded_bgm_dir(create=True), utils.song_dir()):
        for candidate in candidates:
            try:
                return file_security.resolve_path_within_directory(directory, candidate)
            except ValueError as exc:
                last_error = exc
    raise ValueError(str(last_error)) from last_error
