"""Safe reading and writing of persistent files in task directories."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from app.utils import utils


_SCRIPT_LOCKS = tuple(threading.RLock() for _ in range(256))


def _script_lock(task_id: str):
    return _SCRIPT_LOCKS[hash(task_id) % len(_SCRIPT_LOCKS)]


def _script_file(task_id: str) -> Path:
    """Return task script manifest path, reusing unified task directory creation logic."""
    return Path(utils.task_dir(task_id)) / "script.json"


def _write_json_atomic(target: Path, payload: Mapping[str, Any]) -> None:
    """
    Atomically write JSON within target directory to avoid partial files on interruption.

    Temp and target files must reside in the same directory so ``os.replace`` maintains
    atomic replacement semantics across local filesystems and Docker bind mounts.
    Existing files are untouched until success; on error only temp files are cleaned up.
    """
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                payload,
                temp_file,
                ensure_ascii=False,
                indent=4,
                default=lambda value: value.__dict__,
            )
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_script_data(task_id: str, payload: Mapping[str, Any]) -> None:
    """Create or completely replace the task ``script.json`` manifest."""
    with _script_lock(task_id):
        _write_json_atomic(_script_file(task_id), payload)


def patch_script_data(task_id: str, **updates: Any) -> bool:
    """
    Update task manifest while preserving existing fields, returning False on failure.

    Footage source info is diagnostic; disk or permission errors should not block video generation.
    Logs full exception and degrades gracefully; initial creation uses ``write_script_data``.
    """
    try:
        with _script_lock(task_id):
            target = _script_file(task_id)
            with target.open("r", encoding="utf-8") as script_file:
                payload = json.load(script_file)
            if not isinstance(payload, dict):
                raise ValueError("task script data must be a JSON object")

            payload.update(updates)
            _write_json_atomic(target, payload)
            return True
    except FileNotFoundError:
        # ``download_videos`` can be invoked independently without a manifest;
        # do not warn or create partial files for standalone invocations.
        logger.debug(
            f"skip task script update because script.json does not exist: "
            f"task_id={task_id}"
        )
        return False
    except Exception as exc:
        logger.warning(
            "failed to update task script data: "
            f"task_id={task_id}, fields={sorted(updates)}, "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return False
