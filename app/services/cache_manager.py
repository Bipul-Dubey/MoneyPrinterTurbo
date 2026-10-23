"""Statistics, preview, and cleanup service for video material cache."""

from __future__ import annotations

import os
import re
import stat
import time
from dataclasses import dataclass
from typing import Iterator

from loguru import logger

from app.utils import utils


# Online materials use MD5 of URL as stable filenames. Cache manager accepts this format only,
# preventing accidental deletion of user videos or documentation placed in the directory.
_VIDEO_CACHE_FILE_PATTERN = re.compile(r"^vid-[0-9a-f]{32}\.mp4$")
# save_video writes to this same-directory name before atomically publishing a
# validated clip. A forced process exit can leave one behind, so include only
# aged temporary files in cache cleanup and never touch an active download.
_VIDEO_CACHE_TEMP_FILE_PATTERN = re.compile(
    r"^\.vid-[0-9a-f]{32}-[a-z0-9_]+\.mp4$"
)
_SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class VideoCacheStats:
    """Lightweight cache statistics containing filesystem metadata only."""

    file_count: int = 0
    total_size: int = 0
    oldest_mtime: float | None = None
    newest_mtime: float | None = None


@dataclass(frozen=True)
class VideoCacheCleanupResult:
    """Cleanup execution result, tolerating partial file deletion failures."""

    deleted_count: int = 0
    deleted_size: int = 0
    failed_count: int = 0


@dataclass(frozen=True)
class _VideoCacheEntry:
    """Minimal file metadata stored during scan phase without opening or parsing videos."""

    path: str
    name: str
    size: int
    mtime: float
    device: int
    inode: int
    mtime_ns: int


def video_cache_dir() -> str:
    """Return default video cache directory managed by the project."""

    return os.path.realpath(utils.storage_dir("cache_videos"))


def _iter_video_cache_entries(include_temp: bool = False) -> Iterator[_VideoCacheEntry]:
    """
    Sequentially scan top level of default cache directory.

    Uses ``os.scandir`` to reuse directory metadata when cache contains tens of thousands of files.
    Non-recursive, does not open videos or invoke FFmpeg; duration scales linearly with file count.
    """

    cache_dir = video_cache_dir()
    try:
        entries = os.scandir(cache_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning(
            f"failed to scan video cache directory: path={cache_dir}, error={exc}"
        )
        return

    with entries:
        for entry in entries:
            if not _VIDEO_CACHE_FILE_PATTERN.fullmatch(entry.name) and not (
                include_temp and _VIDEO_CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name)
            ):
                continue

            try:
                # Do not follow symlinks, ensuring cleanup stays within default cache boundaries.
                if not entry.is_file(follow_symlinks=False):
                    continue
                # DirEntry.stat() reports st_dev/st_ino as 0 on Windows, which
                # would never match the os.stat() identity check before deletion.
                stat_result = os.stat(entry.path, follow_symlinks=False)
            except OSError as exc:
                logger.warning(
                    f"failed to inspect video cache file: file={entry.name}, error={exc}"
                )
                continue

            yield _VideoCacheEntry(
                path=entry.path,
                name=entry.name,
                size=stat_result.st_size,
                mtime=stat_result.st_mtime,
                device=stat_result.st_dev,
                inode=stat_result.st_ino,
                mtime_ns=stat_result.st_mtime_ns,
            )


def _is_cleanup_candidate(
    entry: _VideoCacheEntry,
    max_age_days: int | None,
    now: float,
) -> bool:
    if (
        _VIDEO_CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name)
        and entry.mtime >= now - _SECONDS_PER_DAY
    ):
        return False
    if max_age_days is None:
        return True
    return entry.mtime < now - max_age_days * _SECONDS_PER_DAY


def _validate_max_age_days(max_age_days: int | None) -> None:
    """Reject invalid cleanup parameters consistently even when cache directory is empty."""
    if max_age_days is None:
        return
    if (
        isinstance(max_age_days, bool)
        or not isinstance(max_age_days, int)
        or max_age_days <= 0
    ):
        raise ValueError("max_age_days must be a positive integer or None")


def get_video_cache_stats(max_age_days: int | None = None) -> VideoCacheStats:
    """
    Summarize all cache or preview cleanup candidates older than specified days.

    ``max_age_days=None`` targets all cache. Reads file size and mtime without reading video content.
    """

    _validate_max_age_days(max_age_days)
    now = time.time()
    file_count = 0
    total_size = 0
    oldest_mtime = None
    newest_mtime = None

    for entry in _iter_video_cache_entries(include_temp=True):
        if not _is_cleanup_candidate(entry, max_age_days, now):
            continue
        file_count += 1
        total_size += entry.size
        oldest_mtime = (
            entry.mtime if oldest_mtime is None else min(oldest_mtime, entry.mtime)
        )
        newest_mtime = (
            entry.mtime if newest_mtime is None else max(newest_mtime, entry.mtime)
        )

    return VideoCacheStats(
        file_count=file_count,
        total_size=total_size,
        oldest_mtime=oldest_mtime,
        newest_mtime=newest_mtime,
    )


def clean_video_cache(max_age_days: int | None = None) -> VideoCacheCleanupResult:
    """
    Clean default video cache and return user-facing summary.

    Re-scans during execution rather than reusing stale previews.
    File deletion is fault-tolerant: individual file errors do not abort the run. Logs
    warnings and continues when files are in use or permissions are insufficient, preventing
    a single abnormal file among hundreds from failing the entire cleanup.
    """

    _validate_max_age_days(max_age_days)
    now = time.time()
    logger.info(
        f"start cleaning video cache: max_age_days={max_age_days}"
    )

    candidate_count = 0
    candidate_size = 0
    deleted_count = 0
    deleted_size = 0
    failed_count = 0
    cache_dir = video_cache_dir()

    # Delete while scanning without keeping the full candidate list in memory. Even if
    # the directory grows to hundreds of thousands of files, cleanup memory overhead
    # remains constant; uses a unified 'now' during execution to prevent cutoff drift.
    for entry in _iter_video_cache_entries(include_temp=True):
        if not _is_cleanup_candidate(entry, max_age_days, now):
            continue
        candidate_count += 1
        candidate_size += entry.size
        try:
            # entry.path comes from the top-level scandir of default dir; re-validate parent
            # directory and filename before deletion to prevent unintended scope expansion.
            if (
                os.path.realpath(os.path.dirname(entry.path)) != cache_dir
                or not (
                    _VIDEO_CACHE_FILE_PATTERN.fullmatch(entry.name)
                    or _VIDEO_CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name)
                )
                or os.path.islink(entry.path)
            ):
                raise ValueError("cache file is outside the managed directory")
            try:
                current = os.stat(entry.path, follow_symlinks=False)
            except FileNotFoundError:
                # Another cleanup already removed the scanned candidate.
                continue
            if not stat.S_ISREG(current.st_mode) or (
                current.st_dev,
                current.st_ino,
                current.st_mtime_ns,
                current.st_size,
            ) != (entry.device, entry.inode, entry.mtime_ns, entry.size):
                # A downloader may atomically publish a new file at this name
                # after the scan. Do not delete the fresh replacement.
                continue
            os.unlink(entry.path)
            deleted_count += 1
            deleted_size += entry.size
        except (OSError, ValueError) as exc:
            failed_count += 1
            logger.warning(
                f"failed to delete video cache file: file={entry.name}, error={exc}"
            )

    logger.info(
        "finished cleaning video cache: "
        f"candidates={candidate_count}, candidate_bytes={candidate_size}, "
        f"deleted={deleted_count}, deleted_bytes={deleted_size}, failed={failed_count}"
    )
    return VideoCacheCleanupResult(
        deleted_count=deleted_count,
        deleted_size=deleted_size,
        failed_count=failed_count,
    )
