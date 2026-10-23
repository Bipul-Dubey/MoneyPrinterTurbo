import copy
import errno
import os
import shutil
import socket
import tempfile
import threading
from contextlib import contextmanager

import toml
from loguru import logger

from app import __version__, paths

root_dir = str(paths.runtime_root())
config_file = f"{root_dir}/config.toml"
_CONTAINER_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "libpod", "podman")
_DOCKER_HOST_GATEWAY_NAME = "host.docker.internal"
_config_save_lock = threading.RLock()
_diagnostics_update_lock = threading.Lock()
_pending_config_lock = threading.RLock()
_pending_config_updates = {}
_pending_config_save_requested = False
_pending_config_flush_scheduled = False
_MISSING = object()
_DELETE = object()
_UTF8_BOM = "\ufeff"


class _SynchronizedConfig(dict):
    """Preserve dict usage syntax while ensuring runtime config writes respect the same lock."""

    def __setitem__(self, key, value):
        # On each full-page rerun, Streamlit rewrites current widget values to config.
        # When a video task holds runtime_config_lock and values have not changed, this write
        # has no side effects and should not freeze the page. Writes that actually change values
        # acquire the lock below, preventing switching Providers, keys, or global settings mid-generation.
        current = super().get(key, _MISSING)
        if current is not _MISSING and current == value:
            return
        with _config_save_lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with _config_save_lock:
            super().__delitem__(key)

    def clear(self):
        if not self:
            return
        with _config_save_lock:
            super().clear()

    def pop(self, key, default=_MISSING):
        # ``pop(key, default)`` also causes no config change if the key does not exist.
        # The WebUI uses this pattern to express 'use default policy', which must complete directly on reruns.
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        with _config_save_lock:
            if default is _MISSING:
                return super().pop(key)
            return super().pop(key, default)

    def setdefault(self, key, default=None):
        # Like __setitem__, setdefault on an existing key is read-only. Returning early
        # allows page reruns that only read default config to proceed without being blocked by long-running task locks.
        current = super().get(key, _MISSING)
        if current is not _MISSING:
            return current
        with _config_save_lock:
            return super().setdefault(key, default)

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        if all(
            (current := dict.get(self, key, _MISSING)) is not _MISSING
            and current == value
            for key, value in changes.items()
        ):
            return
        with _config_save_lock:
            super().update(changes)


def _pending_update_key(config_section, key):
    """Generate a pending update key for in-process fixed config partitions."""
    return id(config_section), key


def update_config_nonblocking(config_section, key, value):
    """
    Non-blocking update of WebUI runtime configuration.

    Video generation holds ``runtime_config_lock`` to prevent a task from switching
    Providers, keys, or voice settings mid-run. Streamlit widget updates cannot wait on
    this long task lock, or the browser would freeze. Updates take effect immediately when
    idle; when busy, only the newest value per option is queued and applied when the lock is released.

    Returns True if the value took effect immediately, False if queued.
    """
    return update_config_values_nonblocking(config_section, {key: value})


def update_config_values_nonblocking(config_section, changes):
    """Queue related fields from the same operation together to prevent split updates under concurrency."""
    # All updates enter the same queue before attempting to acquire the config lock.
    # When multiple pages modify the same setting, queue order determines the final state,
    # preventing an earlier thread from overwriting a newer thread's queued value.
    with _pending_config_lock:
        for key, value in changes.items():
            _pending_config_updates[_pending_update_key(config_section, key)] = (
                config_section,
                key,
                copy.deepcopy(value),
            )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        # Callers typically request save at the end of the Streamlit rerun, but this
        # cannot be guaranteed. For example, if an unhandled exception occurs or updates happen
        # right as a task exits, a background flush thread ensures queued updates eventually persist.
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return all(
            config_section.get(key, _MISSING) == value
            for key, value in changes.items()
        )
    finally:
        _config_save_lock.release()


def update_logging_options(level, log_requests):
    """Serially publish runtime snapshot and persistence intent without waiting on long generation task locks."""
    from app.utils.logging_utils import get_logging_options, set_logging_options

    with _diagnostics_update_lock:
        set_logging_options(level, log_requests)
        level, log_requests = get_logging_options()
        return update_config_values_nonblocking(
            diagnostics, {"log_level": level, "log_requests": log_requests}
        )


def delete_config_nonblocking(config_section, key):
    """
    Non-blocking deletion of WebUI configuration items.

    'Use default value' requires actually removing the config item rather than writing an empty string.
    When a video task holds the lock, deletion overrides any previously queued update for that item
    and takes effect after the task completes.
    """
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            _DELETE,
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return key not in config_section
    finally:
        _config_save_lock.release()


def _apply_pending_config_updates_locked():
    """Apply the newest staged WebUI config values while holding the config write lock."""
    with _pending_config_lock:
        updates = list(_pending_config_updates.values())
        _pending_config_updates.clear()
        # Keep holding the pending lock while applying. Threads reading the 'current + pending' snapshot
        # will see either the complete before or after state, never a partially updated set.
        for config_section, key, value in updates:
            if value is _DELETE:
                config_section.pop(key, None)
            else:
                config_section[key] = value
    return bool(updates)


def snapshot_config_with_pending(config_section):
    """
    Return an effective snapshot of the config partition merged with pending WebUI updates.

    While a video task holds the lock, global configuration cannot be overwritten, but users can still
    prepare their next task. LLM requests using this snapshot will reflect newly selected Provider, model,
    and key settings without altering the running video task.
    """
    with _pending_config_lock:
        snapshot = dict(config_section)
        section_id = id(config_section)
        for (pending_section_id, key), (_, _, value) in _pending_config_updates.items():
            if pending_section_id != section_id:
                continue
            if value is _DELETE:
                snapshot.pop(key, None)
            else:
                snapshot[key] = copy.deepcopy(value)
    return snapshot


def _flush_pending_config_locked(*, suppress_save_errors):
    """Apply and persist all current pending configuration updates while holding the write lock."""
    global _pending_config_save_requested

    updates_applied = _apply_pending_config_updates_locked()
    with _pending_config_lock:
        save_requested = _pending_config_save_requested
        _pending_config_save_requested = False

    if not updates_applied and not save_requested:
        return True

    try:
        save_config()
        return True
    except Exception as exc:
        # In-memory config is successfully applied; on save failure, retain the pending save flag.
        # Video tasks should not fail because config.toml was temporarily unwritable; the next page
        # interaction will retry saving.
        with _pending_config_lock:
            _pending_config_save_requested = True
        if not suppress_save_errors:
            raise
        logger.exception(f"failed to save deferred runtime config: {exc}")
        return False


def _run_deferred_config_flush():
    """Wait for long tasks to release the config lock, then reliably apply and save accumulated updates."""
    global _pending_config_flush_scheduled

    while True:
        with _config_save_lock:
            flush_succeeded = _flush_pending_config_locked(
                suppress_save_errors=True
            )

        with _pending_config_lock:
            has_pending_work = bool(
                _pending_config_updates or _pending_config_save_requested
            )
            if not flush_succeeded or not has_pending_work:
                _pending_config_flush_scheduled = False
                return


def _schedule_deferred_config_flush():
    """Ensure at most one background thread waits to flush configuration at any given time."""
    global _pending_config_flush_scheduled

    with _pending_config_lock:
        if _pending_config_flush_scheduled:
            return
        _pending_config_flush_scheduled = True

    try:
        threading.Thread(
            target=_run_deferred_config_flush,
            name="mpt-config-flush",
            daemon=True,
        ).start()
    except Exception:
        # No worker owns this reservation when construction/start fails.
        # Leave queued updates and the save request intact so a later call
        # can schedule another flush rather than silently waiting forever.
        with _pending_config_lock:
            _pending_config_flush_scheduled = False
        raise


def try_save_config():
    """
    Non-blocking save of WebUI configuration; defers saving until after running long tasks finish if busy.

    Standard API, CLI, and scripts can still call ``save_config`` for blocking write semantics;
    only Streamlit reruns use this function to prevent pages from freezing while waiting on video tasks.
    """
    global _pending_config_save_requested

    with _pending_config_lock:
        _pending_config_save_requested = True

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        return _flush_pending_config_locked(suppress_save_errors=False)
    finally:
        _config_save_lock.release()


@contextmanager
def runtime_config_lock():
    """
    Prevent other WebUI sessions from mutating configuration during an operation dependent on global config.

    The project binds to loopback by default and uses single-user global config. This lightweight lock protects
    long operations like generation or previews from having their Provider or keys changed mid-operation by another tab.
    """
    with _config_save_lock:
        # If the background flush thread had not run when the previous short operation released the lock,
        # the new task must apply queued updates before reading global configuration.
        _flush_pending_config_locked(suppress_save_errors=True)
        try:
            yield
        finally:
            _flush_pending_config_locked(suppress_save_errors=True)


@contextmanager
def try_runtime_config_lock():
    """
    Attempt to acquire runtime config lock and return immediately whether successful.

    WebUI voice preview is a short user-triggered action and should not block for minutes if a background task holds the lock.
    Callers can notify the user to retry; on success, Provider, key, and model settings are guaranteed stable during the preview.
    """
    acquired = _config_save_lock.acquire(blocking=False)
    try:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
        yield acquired
    finally:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
            _config_save_lock.release()


def is_running_in_container(
    dockerenv_path: str = "/.dockerenv",
    containerenv_path: str = "/run/.containerenv",
    cgroup_path: str = "/proc/1/cgroup",
) -> bool:
    """
    Determine whether current process is running inside a container.

    Used primarily for Ollama default Base URL selection:
    - Normal host runtime: `localhost` points to the user machine itself;
    - Inside Docker container: `localhost` points to the container itself; host Ollama
      typically requires `host.docker.internal`.

    Checking `/proc/1/cgroup` existence alone is insufficient as normal Linux systems also have it.
    Returns True only when explicit container markers are detected.
    Path arguments are injectable for unit testing across environments.
    """
    if os.path.isfile(dockerenv_path) or os.path.isfile(containerenv_path):
        return True

    try:
        with open(cgroup_path, mode="r", encoding="utf-8") as fp:
            cgroup_content = fp.read().lower()
    except OSError:
        return False

    return any(marker in cgroup_content for marker in _CONTAINER_CGROUP_MARKERS)


def _can_resolve_hostname(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
    except OSError:
        return False
    return True


def _decode_linux_route_gateway(hex_gateway: str) -> str:
    # The Gateway in /proc/net/route is little-endian hex (e.g. 010011AC is 172.17.0.1).
    # Parsed here so native Linux Docker without host.docker.internal DNS can still reach the host via default gateway.
    if len(hex_gateway) != 8:
        raise ValueError("invalid gateway length")

    octets = [
        str(int(hex_gateway[index : index + 2], 16)) for index in range(6, -1, -2)
    ]
    return ".".join(octets)


def get_container_default_gateway_ip(route_path: str = "/proc/net/route") -> str:
    """
    Read default gateway IP inside a Linux container.

    Docker Desktop provides `host.docker.internal`, but native Linux Docker does not always provide it.
    The default gateway serves as a fallback host address; if Ollama only listens on 127.0.0.1,
    the user still needs to bind Ollama to the host interface or configure `ollama_base_url`.
    """
    try:
        with open(route_path, mode="r", encoding="utf-8") as fp:
            route_lines = fp.readlines()
    except OSError:
        return ""

    for line in route_lines[1:]:
        fields = line.strip().split()
        if len(fields) < 8:
            continue

        destination = fields[1]
        gateway = fields[2]
        if (
            destination != "00000000"
            or fields[7] != "00000000"
            or gateway == "00000000"
        ):
            continue

        try:
            return _decode_linux_route_gateway(gateway)
        except ValueError:
            logger.warning(f"invalid container gateway route entry: {line.strip()}")
            return ""

    return ""


def get_default_ollama_base_url() -> str:
    """
    Return default OpenAI-compatible base_url for Ollama.

    Not used when `ollama_base_url` is explicitly configured; handles optimal default when unset.
    Container environments default to host; normal local execution defaults to localhost.
    """
    if not is_running_in_container():
        return "http://localhost:11434/v1"

    if _can_resolve_hostname(_DOCKER_HOST_GATEWAY_NAME):
        return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"

    gateway_ip = get_container_default_gateway_ip()
    if gateway_ip:
        logger.info(
            "host.docker.internal is not resolvable, fallback to container "
            f"default gateway for Ollama: {gateway_ip}"
        )
        return f"http://{gateway_ip}:11434/v1"

    logger.warning(
        "failed to resolve host.docker.internal and container default gateway; "
        "fallback to host.docker.internal for Ollama"
    )
    return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"


def _load_toml_config(config_path: str):
    """
    Load TOML with compatibility for duplicate UTF-8 BOMs written by some Windows editors.

    ``utf-8-sig`` only strips one BOM at the file start. Certain Windows editors or extraction
    tools write multiple BOMs, causing invisible characters in the TOML parser on line 1.
    Performs a read-only normalization on parse error without rewriting the file to avoid overwriting keys.
    """
    try:
        return toml.load(config_path)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "load config failed, retry with UTF-8 BOM compatibility: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )

    try:
        with open(config_path, mode="r", encoding="utf-8-sig") as fp:
            config_content = fp.read()

        normalized_content = config_content.lstrip(_UTF8_BOM)
        removed_bom_count = len(config_content) - len(normalized_content)
        if removed_bom_count:
            logger.warning(
                "removed repeated UTF-8 BOM characters while loading config: "
                f"path={config_path}, count={removed_bom_count}"
            )
        return toml.loads(normalized_content)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.error(
            "config file is not valid TOML after UTF-8 BOM normalization: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )
        raise



def _initialize_config_from_example(example_file):
    """Publish a complete first-boot config without replacing another writer."""
    fd, temp_path = tempfile.mkstemp(
        prefix=".config-init-", suffix=".toml.tmp",
        dir=os.path.dirname(os.path.abspath(config_file)),
    )
    try:
        with os.fdopen(fd, "wb") as destination:
            with open(example_file, "rb") as source:
                shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            if os.name == "nt":
                # Windows rename refuses an existing destination. POSIX rename
                # overwrites it, so use an exclusive hard-link publication there.
                os.rename(temp_path, config_file)
            else:
                os.link(temp_path, config_file)
        except FileExistsError:
            return False
        except OSError as exc:
            raise OSError(
                exc.errno,
                "cannot safely initialize config.toml; copy config.example.toml "
                "to config.toml before starting the application",
            ) from exc
        return True
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            # A successful Windows rename consumed the temporary path.
            pass


def load_config():
    os.makedirs(root_dir, mode=0o700, exist_ok=True)
    # Docker may create an empty directory when a missing config.toml is bind-mounted.
    # Only remove that empty placeholder; a nonempty directory may contain user data.
    if os.path.isdir(config_file):
        try:
            os.rmdir(config_file)
        except FileNotFoundError:
            # Another initializer already removed this empty bind-mount stub.
            pass
        except OSError as exc:
            # Another initializer may already have replaced the empty Docker
            # stub with its complete regular file. Accept only that race, not
            # permission failures or a still-present/nonempty directory.
            if not (
                isinstance(exc, NotADirectoryError) and os.path.isfile(config_file)
            ):
                raise IsADirectoryError(
                    f"{config_file} is a directory and cannot be used as the config file; "
                    "move or rename it, then restart"
                ) from exc

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if not os.path.isfile(example_file):
            example_file = str(paths.config_template())
        if os.path.isfile(example_file):
            if _initialize_config_from_example(example_file):
                logger.info("copy config.example.toml to config.toml")

    logger.info(f"load config from file: {config_file}")

    return _load_toml_config(config_file)


def save_config():
    """
    Atomically save runtime configuration.

    Different Streamlit sessions may trigger saves concurrently. Overwriting config.toml directly
    can cause another thread to read partial TOML content. Serializes saving via in-process reentrant lock,
    writes to a sibling temporary file, and atomically replaces the target via os.replace.

    Single-file bind mounts on Docker Desktop make config.toml a mount point, where the Linux kernel
    forbids replacing the mount point via rename/replace (returning EBUSY). In that case, it falls back
    to in-place overwrite under the lock; other exceptions are re-raised to surface permission or disk errors.

    Preserves single-user global config semantics without introducing complex multi-user overhead,
    preventing file corruption across multiple tabs or fast reruns.
    """
    with _config_save_lock:
        config_to_save = dict(_cfg)
        config_to_save["app"] = dict(app)
        config_to_save["azure"] = dict(azure)
        config_to_save["siliconflow"] = dict(siliconflow)
        config_to_save["minimax_tts"] = dict(minimax_tts)
        config_to_save["elevenlabs"] = dict(elevenlabs)
        config_to_save["chatterbox"] = dict(chatterbox)
        config_to_save["kokoro"] = dict(kokoro)
        config_to_save["fish_audio"] = dict(fish_audio)
        config_to_save["voxcpm"] = dict(voxcpm)
        config_to_save["ui"] = dict(ui)
        config_to_save["diagnostics"] = dict(diagnostics)
        serialized_config = toml.dumps(config_to_save)

        # Saved at the end of a full WebUI rerun. Return directly if content hasn't changed
        # to avoid unnecessary disk writes and fsync on every widget click.
        try:
            with open(config_file, mode="r", encoding="utf-8") as f:
                if f.read() == serialized_config:
                    _cfg.clear()
                    _cfg.update(config_to_save)
                    return
        except (OSError, UnicodeError):
            pass

        temp_path = ""
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=".config-",
                suffix=".toml.tmp",
                dir=root_dir,
            )
            with os.fdopen(fd, mode="w", encoding="utf-8") as f:
                f.write(serialized_config)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(temp_path, config_file)
            except OSError as exc:
                if exc.errno != errno.EBUSY:
                    raise

                logger.warning(
                    "atomic config replacement is unavailable for the mounted "
                    f"file, fallback to in-place write: {config_file}"
                )
                with open(config_file, mode="w", encoding="utf-8") as f:
                    f.write(serialized_config)
                    f.flush()
                    os.fsync(f.fileno())
            _cfg.clear()
            _cfg.update(config_to_save)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


_cfg = load_config()
diagnostics = _SynchronizedConfig(_cfg.get("diagnostics", {}))
app = _SynchronizedConfig(_cfg.get("app", {}))
whisper = _cfg.get("whisper", {})
proxy = _cfg.get("proxy", {})
azure = _SynchronizedConfig(_cfg.get("azure", {}))
siliconflow = _SynchronizedConfig(_cfg.get("siliconflow", {}))
minimax_tts = _SynchronizedConfig(_cfg.get("minimax_tts", {}))
elevenlabs = _SynchronizedConfig(_cfg.get("elevenlabs", {}))
chatterbox = _SynchronizedConfig(_cfg.get("chatterbox", {}))
kokoro = _SynchronizedConfig(_cfg.get("kokoro", {}))
fish_audio = _SynchronizedConfig(_cfg.get("fish_audio", {}))
voxcpm = _SynchronizedConfig(_cfg.get("voxcpm", {}))
ui = _SynchronizedConfig(
    _cfg.get(
        "ui",
        {
            "hide_log": False,
        },
    )
)

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "MoneyPrinterTurbo")
project_description = _cfg.get(
    "project_description",
    "<a href='https://github.com/harry0703/MoneyPrinterTurbo'>https://github.com/harry0703/MoneyPrinterTurbo</a>",
)
project_version = _cfg.get("project_version", __version__)
reload_debug = False

app["redis_host"] = os.getenv(
    "MPT_APP_REDIS_HOST",
    os.getenv("REDIS_HOST", app.get("redis_host", "localhost")),
)

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path and os.path.isfile(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger.info(f"{project_name} v{project_version}")
