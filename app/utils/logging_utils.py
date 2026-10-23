import functools
import os
import re
import threading

from loguru import logger


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
_PROJECT_ROOT_IN_MESSAGE_RE = re.compile(re.escape(PROJECT_ROOT) + r"(?![\w.\-])")
LOG_RECORD_FORMAT = (
    "<green>{time:%Y-%m-%d %H:%M:%S}</> | "
    "<level>{level}</> | "
    '"{file.path}:{line}":<blue> {function}</> '
    "- <level>{message}</>\n"
)
# Loguru starts with terminal handler ID 0. On WebUI reload, only replace this
# base terminal output; do not call logger.remove() to clear all handlers, or active
# tasks' temporary sinks collecting WebUI logs would also be removed.
_terminal_handler_id: int | None = 0
_terminal_handler_lock = threading.RLock()
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
# Only replace immutable snapshots; logging threads avoid acquiring generation task locks.
_runtime_logging = ("DEBUG", False)


def get_logging_options() -> tuple[str, bool]:
    return _runtime_logging


def set_logging_options(level: str, log_requests: bool = False) -> None:
    """Adjust in-process logging on the fly without rebuilding handlers or disrupting running tasks."""
    global _runtime_logging
    level = str(level).strip().upper()
    if level not in LOG_LEVELS:
        raise ValueError(f"unsupported log level: {level}")
    if not isinstance(log_requests, bool):
        raise ValueError("log_requests must be a boolean")
    _runtime_logging = (level, log_requests)


def runtime_log_filter(record) -> bool:
    level, log_requests = _runtime_logging
    # Request tracing is an independent flag; users can explicitly enable diagnostics even at ERROR level.
    if record["extra"].get("request_trace"):
        return log_requests
    return record["level"].no >= logger.level(level).no


# WebUI filters logs by worker thread ID. Worker threads for parallel downloads,
# clip encoding, and heartbeats have their own thread IDs whose logs would otherwise be discarded,
# causing long stages to show no output in the WebUI. Track 'helper thread -> parent thread'
# mapping so filters attribute helper thread records back to the owning task.
_log_scope_roots: dict[int, int] = {}
_log_scope_lock = threading.Lock()


def log_scope_thread_id(thread_id: int | None = None) -> int:
    """Return the root thread ID for a thread's log scope; unbound threads return themselves."""
    if thread_id is None:
        thread_id = threading.get_ident()
    with _log_scope_lock:
        return _log_scope_roots.get(thread_id, thread_id)


def bind_log_scope(func):
    """
    Wrap a function to execute in another thread, attributing its logs to the caller's thread.

    Must be called from the task-submitting thread: scope is fixed upon wrapping, and any further
    threads started by helper threads map back to the root task thread. Unbinds on exit because
    thread IDs are reused by the system, preventing leftover mappings from attributing unrelated future logs.
    """
    root_thread_id = log_scope_thread_id()

    @functools.wraps(func)
    def run_in_log_scope(*args, **kwargs):
        thread_id = threading.get_ident()
        if thread_id == root_thread_id:
            return func(*args, **kwargs)

        with _log_scope_lock:
            previous_root = _log_scope_roots.get(thread_id)
            _log_scope_roots[thread_id] = root_thread_id
        try:
            return func(*args, **kwargs)
        finally:
            with _log_scope_lock:
                # Thread pools reuse threads: restore previous ownership rather than deleting outright,
                # preventing nested wrappers from prematurely clearing an outer active binding.
                if previous_root is None:
                    _log_scope_roots.pop(thread_id, None)
                else:
                    _log_scope_roots[thread_id] = previous_root

    return run_in_log_scope


def _project_relative_path(file_path):
    """
    Shorten absolute paths to forward-slash project-relative paths starting with ``./``.

    On Windows, projects may start from mapped network drives or ``subst`` drives. The stack frame
    path is ``X:\MoneyPrinterTurbo\...`` while ``PROJECT_ROOT`` resolves via ``realpath`` to ``C:\...``,
    causing ``os.path.relpath`` to raise ``ValueError``. If formatting fails, loguru drops the record,
    emptying both terminal and WebUI panels; fall back to the raw path. Outside project files follow
    the same principle: prefixing ``./`` onto ``..`` paths only creates confusing output.
    """
    try:
        relative_path = os.path.relpath(file_path, PROJECT_ROOT)
    except ValueError:
        return file_path
    if relative_path == os.pardir or relative_path.startswith(os.pardir + os.sep):
        return file_path
    # Windows relpath returns backslashes; naive concatenation produces mixed separators like ``./app\utils``
    # inconsistent with other platforms.
    return f"./{relative_path.replace(os.sep, '/')}"


def format_log_record(record):
    """
    Uniformly format terminal and WebUI logs.

    Loguru delivers each record to multiple sinks. The first sink may have already converted
    absolute paths to project-relative paths, so both forms are supported.
    The WebUI sink disables ANSI colors, while timestamps, levels, locations, and messages remain identical.
    """
    file_path = record["file"].path
    if os.path.isabs(file_path):
        record["file"].path = _project_relative_path(file_path)

    # Log messages sometimes contain absolute paths to task files. Shortening them to project-relative
    # paths avoids discrepancies between WebUI and terminal.
    # Match the full project root path only; sibling directories with identical prefixes like `<root>-backup`
    # must not be corrupted into `.-backup`.
    record["message"] = _PROJECT_ROOT_IN_MESSAGE_RE.sub(".", record["message"])
    return LOG_RECORD_FORMAT


def configure_terminal_logger(sink, level: str, colorize: bool = True) -> int:
    """
    Safely replace process-level terminal log handler while preserving task-specific handlers.

    Streamlit re-executes logging setup on hot reload or cache expiry. Precisely remove the old terminal
    output by stored handler ID without interrupting WebUI logs being written by background tasks.
    A lock protects handler ID updates under concurrent browser session initializations.
    """
    global _terminal_handler_id

    set_logging_options(level, get_logging_options()[1])

    with _terminal_handler_lock:
        if _terminal_handler_id is not None:
            try:
                logger.remove(_terminal_handler_id)
            except ValueError:
                # Tests or external callers may have removed the handler. Proceed to create
                # a new terminal output without affecting other valid log sinks.
                pass

        _terminal_handler_id = logger.add(
            sink,
            level=0,
            filter=runtime_log_filter,
            format=format_log_record,
            colorize=colorize,
        )
        return _terminal_handler_id
