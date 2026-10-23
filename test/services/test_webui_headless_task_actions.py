import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config

# When webui/Main.py imports app.services.material, it transitively imports moviepy, which resolves
# the FFmpeg executable at import time (depending on sys.platform for platform binaries); numpy also
# decides whether to invoke os.uname() based on sys.platform. These resolutions must complete prior to the
# fixture below; otherwise simulating a headless server on Windows looks for Linux FFmpeg or raises AttributeError.
from app.services import material as _material  # noqa: F401
from app.services import state as sm
from app.utils import utils

ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


@pytest.fixture
def headless_task_app(tmp_path, monkeypatch):
    # Use isolated tasks directory to avoid reading or modifying developer real generation records. Test files only need
    # to be registered by Streamlit as media resources without participating in video decoding; thus invoking FFmpeg to generate
    # media in unit tests is unnecessary, reliably covering headless server UI branches.
    tasks_dir = tmp_path / "storage" / "tasks"
    task_dir = tasks_dir / "headless-test"
    task_dir.mkdir(parents=True)
    video_file = task_dir / "final-1.mp4"
    video_file.write_bytes(b"test video payload")

    monkeypatch.setattr(utils, "task_dir", lambda: str(tasks_dir))
    monkeypatch.setattr(sm.state, "get_all_tasks", lambda *_args, **_kwargs: ([], 0))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    # AppTest re-executes page scripts multiple times; configuration saving must remain isolated
    # throughout the test lifecycle to prevent widget initialization from accidentally writing to developer config.toml.
    with patch.object(config, "try_save_config", return_value=True):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.run()
        yield app, video_file


def _button_by_key_prefix(app, key_prefix):
    return next(button for button in app.button if str(button.key).startswith(key_prefix))


def test_headless_play_renders_and_closes_browser_preview(headless_task_app):
    app, video_file = headless_task_app

    _button_by_key_prefix(app, "play_task_all_headless-test").click()
    app.run()

    assert not app.exception
    assert app.session_state["task_preview_video_file"] == str(video_file.resolve())
    assert len(app.get("video")) == 1

    _button_by_key_prefix(app, "close_task_video_preview").click()
    app.run()

    assert "task_preview_video_file" not in app.session_state
    assert len(app.get("video")) == 0


def test_headless_open_folder_shows_host_mapped_path(headless_task_app):
    app, _ = headless_task_app

    _button_by_key_prefix(app, "open_task_all_headless-test").click()
    app.run()

    assert not app.exception
    # _open_task_path() uses os.path.relpath() to generate the relative segment, so separators follow the platform
    # running the test. The headless branch is entered via monkeypatched sys.platform, so toast text on Windows still
    # contains backslashes; assertions must not hardcode POSIX separators.
    expected_folder = os.path.join("tasks", "headless-test")
    assert any(
        f"./storage/{expected_folder}" in toast.value for toast in app.get("toast")
    )
