import hashlib
import html
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import requests
import streamlit as st
from loguru import logger
from streamlit_tour import Tour

# When WebUI runs as a standalone entry point, prioritize the project root
# over third-party dependencies to avoid a same-named app package shadowing MoneyPrinterTurbo's own app package.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)

from app.config import config
from app.models import const
from app.models.llm_provider import (
    DEFAULT_LLM_PROVIDER_ID,
    LLM_PROVIDER_REGISTRY,
    get_llm_provider,
    normalize_provider_override,
)
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
from app.services import material_upload as material_upload_service
from app.services import (
    cache_manager,
    llm,
    loomloom,
    material,
    metaso_minimax,
    muapi,
    ofox,
    subtitle,
    video,
    volcengine_seedance,
    voice,
    webui_task,
)
from app.services import elevenlabs_music as elevenlabs_music_service
from app.services import sonilo as sonilo_service
from app.services import state as sm
from app.services import task as tm
from app.services import version_checker
from app.utils.logging_utils import (
    LOG_LEVELS,
    configure_terminal_logger,
    get_logging_options,
)
from app.utils import utils
from app.utils.request_logging import trace_http_call

st.set_page_config(
    page_title="MoneyPrinterTurbo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Report a bug": "https://github.com/harry0703/MoneyPrinterTurbo/issues",
        "About": "# MoneyPrinterTurbo\nSimply provide a topic or keyword for a video, and it will "
        "automatically generate the video copy, video materials, video subtitles, "
        "and video background music before synthesizing a high-definition short "
        "video.\n\nhttps://github.com/harry0703/MoneyPrinterTurbo",
    },
)


# Streamlit 1.59 displays platform entry points such as Deploy and skills nudge by default in the top right corner.
# MoneyPrinterTurbo is a local tool intended for end users; these entry points create large blank areas at the top
# and may mislead new users into thinking extra components need to be installed. Hide the Streamlit platform toolbar
# and compress top padding of the main container, preserving only the title, language selection, and business settings.
style_file = Path(__file__).with_name("styles.css")
streamlit_style = f"<style>{style_file.read_text(encoding='utf-8')}</style>"
st.markdown(streamlit_style, unsafe_allow_html=True)
# Define resource directories
font_dir = utils.font_dir()
song_dir = utils.song_dir()
i18n_dir = os.path.join(root_dir, "webui", "i18n")
config_file = os.path.join(root_dir, "webui", ".streamlit", "webui.toml")
# The locale list must be available before session state initialization so the browser locale can be mapped
# to a supported language on first access; auto-detection results only enter the current session without modifying global config.
locales = utils.load_locales(i18n_dir)
DEFAULT_CHATTERBOX_BASE_URL = "http://127.0.0.1:4123/v1"
DEFAULT_CHATTERBOX_MODEL = "chatterbox"
DEFAULT_CHATTERBOX_VOICES = ["default-Female"]
DEFAULT_KOKORO_BASE_URL = "http://127.0.0.1:8880/v1"
DEFAULT_KOKORO_MODEL = "kokoro"
# empty = ask the server for its voice list (GET {base_url}/audio/voices)
DEFAULT_KOKORO_VOICES: list[str] = []
DEFAULT_VOXCPM_BASE_URL = voice.VOXCPM_DEFAULT_BASE_URL
DEFAULT_VOXCPM_VOICE = voice.VOXCPM_DEFAULT_VOICE
VOXCPM_REFERENCE_AUDIO_SESSION_KEY = "voxcpm_reference_audio"
VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY = "voxcpm_reference_audio_error"
VOXCPM_PROMPT_AUDIO_SESSION_KEY = "voxcpm_prompt_audio"
VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY = "voxcpm_prompt_audio_error"
VOXCPM_PROMPT_TEXT_SESSION_KEY = "voxcpm_prompt_text_input"
VOXCPM_HIGH_FIDELITY_SESSION_KEY = "voxcpm_high_fidelity_enabled"
VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY = "voxcpm_separate_prompt_audio_enabled"
VOXCPM_PROMPT_EXAMPLE_MODE_SESSION_KEY = "voxcpm_prompt_example_mode"
ONBOARDING_TOUR_KEY = "mpt-onboarding-v1"
CUSTOM_LLM_ENDPOINT_ID = "custom"
VOICE_MODE_TTS = "tts"
VOICE_MODE_UPLOAD = "upload"
VOICE_MODE_NONE = "none"
LOOMLOOM_MAX_POLL_FAILURES = 5
# WebUI groups video sources by material capability, while preserving the underlying video_source values.
# The AI video group shares the same business order with the settings page: partner providers first (MetaSo, OFox,
# Sonilo, Volcengine), followed by the remaining services. This ensures consistent ordering across both entry points
# without altering field semantics in config.toml, historical tasks, or API requests, requiring no migration for existing users.
VIDEO_SOURCE_GROUPS = {
    "stock_video": ("pexels", "pixabay", "coverr"),
    "ai_video": (
        "metaso_minimax",
        "ofox",
        "loomloom",
        "volcengine_seedance",
        "wavespeed",
        "muapi",
    ),
    "ai_image": ("openai_image",),
    "local": ("local",),
}
# Upload-Post API keys and publishing users are managed across two separate pages, and publishing usernames
# do not equal login emails. Centralizing these entry points avoids discrepancies from hardcoding URLs in i18n copy
# and makes it convenient for users to complete initial configuration and account maintenance directly from WebUI.
UPLOAD_POST_API_KEYS_URL = "https://app.upload-post.com/api-keys"
UPLOAD_POST_MANAGE_USERS_URL = "https://app.upload-post.com/manage-users"
# Material settings and video source descriptions share a referral URL to avoid inconsistent link parameters across locations.
OFOX_REFERRAL_URL = (
    "https://ofox.ai/?utm_source=github"
    "&utm_medium=sponsorship&utm_content=moneyprinterturbo"
)
# '__default__' is a WebUI-specific sentinel; it is never written to config.toml nor passed to FFmpeg.
# The backend falls back to stable libx264 when video_codec is not configured; retaining this sentinel distinguishes
# 'follow project default policy' from 'user explicitly pinned libx264', facilitating safe future adjustments to the default policy.
DEFAULT_VIDEO_CODEC_OPTION = "__default__"
# LoomLoom's capabilities endpoint only returns model IDs and display names, without pricing. Maintain user-confirmed
# reference prices here to assist in model selection; final billing is settled based on actual model calls. Aliases cover
# both current display names and common model IDs; unlisted new models return empty prices without affecting selection or quotes.
LOOMLOOM_VIDEO_MODEL_PRICES = (
    (("veo31fast", "googleveo31fastpreview"), "￥0.700/秒", "￥0.700/秒"),
    (
        (
            "通义万相22图生视频fastlora",
            "通义万相22文生视频fastlora",
            "tongyiwanxiang22i2vfastlora",
            "tongyiwanxiang22t2vfastlora",
            "wanx22i2vfastlora",
            "wanx22t2vfastlora",
        ),
        "￥0.350–0.770/条",
        "￥0.350/条（480P）；￥0.770/条（720P）",
    ),
    (("即梦30文生视频720p", "jimeng30t2v720p"), "￥0.230/秒", "￥0.230/秒"),
    (("即梦30pro视频", "jimeng30pro视频", "jimeng30provideo"), "￥1.000/秒", "￥1.000/秒"),
    (("veo3", "googleveo3"), "￥1.400/秒", "￥1.400/秒"),
    (("veo31", "googleveo31"), "￥1.400/秒", "￥1.400/秒"),
    (
        ("klingv2", "可灵v2"),
        "￥10.00–20.00/条",
        "￥10.00/条（5 秒）；￥20.00/条（10 秒）",
    ),
    (
        ("klingv21master", "可灵v21master"),
        "￥10.00–20.00/条",
        "￥10.00/条（5 秒）；￥20.00/条（10 秒）",
    ),
    (
        ("viduq3pro",),
        "￥0.440–1.000/秒",
        "￥0.440/秒（540P）；￥0.940/秒（720P）；￥1.000/秒（1080P）",
    ),
)
DEFAULT_SUBTITLE_SETTINGS = {
    "subtitle_enabled": True,
    "font_name": "MicrosoftYaHeiBold.ttc",
    "subtitle_position": "bottom",
    "subtitle_display_mode": "sentence",
    "subtitle_animation": "none",
    "custom_position": 70.0,
    "text_fore_color": "#FFFFFF",
    "font_size": 60,
    "stroke_color": "#000000",
    "stroke_width": 1.5,
    "subtitle_background_enabled": False,
    "subtitle_background_color": "#000000",
    "rounded_subtitle_background": False,
}
LOCAL_MATERIAL_EXTENSIONS = set(material_upload_service.SUPPORTED_MATERIAL_EXTENSIONS)
CUSTOM_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
_FINAL_VIDEO_PATTERN = re.compile(
    r"^final-(?P<index>\d+)\.(?P<extension>mp4|mov|mkv|webm)$",
    re.IGNORECASE,
)
# Share the same upload filename rule list with app/services/bgm.py and app/services/material_upload.py:
# In addition to path separators and Win32 invalid characters, strip all control characters and glyph-altering characters.
# Previously, only ord < 32 (C0 control characters) was intercepted, missing the other half of the problem:
# * C1 control characters U+007F-U+009F: pass verbatim into download filenames, logs, and UI.
# * Bidirectional text control characters U+200E/U+200F/U+202A-U+202E/U+2066-U+2069: characters after U+202E
# render in reverse order, making ``holiday<U+202E>4pm`` look like a different file extension.
# * U+2028/U+2029 (Unicode line/paragraph separators) create visual newlines within a single log line.
# Video topics are free-form user inputs that may paste these characters, so they must be filtered here.
# Strings avoid the r'' prefix: ``\\`` denotes the backslash itself, while ``\x``/``\u`` escapes are parsed
# by Python to avoid writing escaped text directly into the character class.
_DOWNLOAD_FILENAME_INVALID_PATTERN = re.compile(
    "[<>:\"/\\\\|?*\\x00-\\x1f\\x7f-\\x9f"
    "\\u200e\\u200f\\u2028\\u2029\\u202a-\\u202e\\u2066-\\u2069]"
)
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {
        f"{prefix}{number}"
        for prefix in ("COM", "LPT")
        for number in range(1, 10)
    }
    # Win32 also recognizes Latin-1 superscript digits ¹, ², ³ as device numbers. While rare in topics,
    # they still cause download failures on Windows, so they are handled alongside regular numeric reserved names.
    | {
        f"{prefix}{number}"
        for prefix in ("COM", "LPT")
        for number in ("¹", "²", "³")
    }
)
_RUNTIME_CONFIG_SECTIONS = {
    "app": config.app,
    "azure": config.azure,
    "chatterbox": config.chatterbox,
    "kokoro": config.kokoro,
    "elevenlabs": config.elevenlabs,
    "minimax_tts": config.minimax_tts,
    "siliconflow": config.siliconflow,
    "fish_audio": config.fish_audio,
    "voxcpm": config.voxcpm,
    "ui": config.ui,
}
# Settings presets and key backups use distinct file identifiers. Validate schema and version upon import
# to prevent misidentifying task logs, config.toml, or other JSON files as export files for this feature.
SETTINGS_PRESET_SCHEMA = "moneyprinterturbo.settings-preset"
SETTINGS_PRESET_VERSION = 1
SETTINGS_PRESET_FILE_NAME = "moneyprinterturbo-settings.json"
KEY_BACKUP_SCHEMA = "moneyprinterturbo.key-backup"
KEY_BACKUP_VERSION = 1
KEY_BACKUP_FILE_NAME = "moneyprinterturbo-keys.json"
# Export files contain only settings or credentials, not media. Reject oversized
# uploads before decoding and parsing them in the Streamlit process.
MAX_SETTINGS_TRANSFER_BYTES = 2 * 1024 * 1024
# Presets only describe generation parameters. Materials, voiceovers, and background music are local file paths;
# presets are typically imported on another machine or container, where including these paths would point to nonexistent files.
PRESET_EXCLUDED_PARAM_KEYS = frozenset(
    {
        "video_materials",
        "custom_audio_file",
        "bgm_file",
    }
)
# Keys are identified by configuration key name suffixes. Newly added providers automatically enter backups
# as long as they follow the existing naming convention, eliminating the need to maintain a separate key list.
CREDENTIAL_KEY_SUFFIXES = (
    "api_key",
    "api_keys",
    "api_token",
    "access_key",
    "secret_key",
    "speech_key",
)
# Restoring keys without their accompanying configuration items leaves credentials unusable; back them up together.
CREDENTIAL_COMPANION_KEYS = {
    # Azure Voice requires knowing the region at the same time.
    "azure": ("speech_region",),
    # Provider extra fields are declared by the Registry, such as Account ID and Gateway ID for Cloudflare AI Gateway.
    # Restoring only the API Key while dropping these fields prevents the provider from being called on another machine.
    # Reading from Registry allows newly added providers to enter backups automatically without maintaining a second field list here.
    "app": tuple(
        provider.config_key(field.config_suffix)
        for provider in LLM_PROVIDER_REGISTRY
        for field in provider.extra_fields
    ),
}

NON_LLM_COMPANION_KEYS = {
    "app": ("upload_post_username",)
}
# The same key may use different widget keys across panels: the audio panel directly edits Gemini and
# MiMo LLM keys. Clear every alias during backup restoration; otherwise stale leftover values will overwrite
# the restored keys on the next rerun.
CREDENTIAL_WIDGET_STATE_ALIASES = {
    ("app", "gemini_api_key"): ("gemini_tts_api_key_input",),
    ("app", "mimo_api_key"): ("mimo_tts_api_key_input",),
}
# The ui section only stores interface preferences and contains no credentials; skip it entirely during backups.
KEY_BACKUP_EXCLUDED_SECTIONS = frozenset({"ui"})


# -----------------------------------------------------------------------------
# Startup configuration, session state, and localization
# -----------------------------------------------------------------------------


def _set_runtime_config(section_name, key, value):
    """
    Update WebUI configuration without waiting for background video generation tasks.

    Before background tasks finish, the config layer only retains the latest value for each item;
    when tasks release the config lock, updates are automatically applied and saved. Widget values
    are still tracked by Streamlit session_state, so intermediate reruns will not reset newly input
    values to stale configuration.
    """
    config_section = _RUNTIME_CONFIG_SECTIONS[section_name]
    updated = config.update_config_nonblocking(config_section, key, value)
    if not updated:
        logger.debug(f"deferred WebUI config update: section={section_name}, key={key}")
    return updated


def _delete_runtime_config(section_name, key):
    """Delete a WebUI configuration item; deferred if background tasks hold the config lock."""
    config_section = _RUNTIME_CONFIG_SECTIONS[section_name]
    deleted = config.delete_config_nonblocking(config_section, key)
    if not deleted:
        logger.debug(f"deferred WebUI config delete: section={section_name}, key={key}")
    return deleted


def _save_runtime_config():
    """Request saving WebUI configuration; returns immediately if background tasks hold the config lock."""
    saved = config.try_save_config()
    if not saved:
        logger.debug("deferred WebUI config save until active task completes")
    return saved


def _saved_ui_choice(key, options, default, section=None):
    """Read a persisted choice, falling back to default on invalid legacy or manually edited values."""
    options = list(options)
    section = config.ui if section is None else section
    saved = section.get(key, default)
    numeric_default = isinstance(default, (int, float)) and not isinstance(
        default, bool
    )
    # bool is a subclass of int, so ``True == 1``. Reject manually edited TOML
    # booleans for numeric options so they cannot masquerade as the first numeric option.
    if numeric_default and isinstance(saved, bool):
        return default
    for option in options:
        if saved == option:
            # Return the real value from options, normalizing TOML 1.0 equivalents
            # to integer option 1 to avoid downstream parameter type drift.
            return option

    # Numeric values in TOML usually keep their original type; still accommodate manually entered strings.
    if numeric_default and isinstance(saved, str):
        try:
            converted = type(default)(saved)
        except (TypeError, ValueError):
            converted = None
        for option in options:
            if converted == option:
                return option
    return default


def _saved_ui_number(key, default, minimum, maximum, number_type=float):
    """Read and clamp persisted numeric values to prevent invalid configurations from breaking Streamlit sliders."""
    try:
        saved = config.ui.get(key, default)
        if isinstance(saved, bool):
            raise ValueError("boolean is not a numeric setting")
        value = number_type(saved)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite value")
    except (TypeError, ValueError, OverflowError):
        value = default
    return min(maximum, max(minimum, value))


def _saved_ui_bool(key, default):
    """Support TOML booleans and common manual strings, rejecting ambiguous legacy values."""
    value = config.ui.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _saved_ui_color(key, default):
    """Pass only standard 6-digit hex colors to the Streamlit color picker."""
    value = str(config.ui.get(key, default) or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return value
    return default


def _saved_ui_text(key, default="", max_length=None):
    """Read persisted text while respecting the max length of the corresponding WebUI widget."""
    value = str(config.ui.get(key, default) or default)
    if max_length is not None:
        value = value[:max_length]
    return value


def _run_llm_read_operation(operation_name, operation):
    """
    Execute read-only requests using the stable current LLM config without waiting for video generation tasks.

    Uses mutex protection when the configuration lock is immediately available; when held by a background
    video task, global configuration remains unchanged until completion, allowing safe cloning of current config
    overlaid with pending provider, model, and key changes from the UI. This ensures new copy uses the latest
    UI selections without mutating running video tasks.
    """
    with config.try_runtime_config_lock() as lock_acquired:
        # The configuration layer holds the queue lock while copying global values and overlaying pending updates,
        # ensuring the snapshot sees only complete states before or after updates without mixing provider parameters.
        app_config_snapshot = config.snapshot_config_with_pending(config.app)
        if lock_acquired:
            return operation(app_config_snapshot)

    logger.info(
        f"run read-only LLM operation with active task configuration: "
        f"operation={operation_name}"
    )
    return operation(app_config_snapshot)


def _parse_chatterbox_voices(voices):
    # Chatterbox is a self-hosted service whose voice list is entered manually in the WebUI.
    # Uniformly accommodate TOML arrays and comma-separated strings to avoid state discrepancies
    # among selectboxes, preview buttons, and subsequent generation pipelines.
    if isinstance(voices, str):
        return [v.strip() for v in voices.split(",") if v.strip()]
    return [str(v).strip() for v in voices or [] if str(v).strip()]


def _sync_chatterbox_config_from_session_state():
    # Streamlit buttons trigger full-page reruns, and the Chatterbox config inputs are located after
    # the 'Preview Voice' button. Reading only config.chatterbox during preview might miss base_url/model/voices
    # freshly entered by the user. Synchronizing once from session_state ensures both button logic
    # and widget display logic use the same latest configuration.
    _set_runtime_config(
        "chatterbox",
        "base_url",
        (
            st.session_state.get(
                "chatterbox_base_url_input",
                config.chatterbox.get("base_url") or DEFAULT_CHATTERBOX_BASE_URL,
            )
            or ""
        ).strip(),
    )
    _set_runtime_config(
        "chatterbox",
        "api_key",
        st.session_state.get(
            "chatterbox_api_key_input", config.chatterbox.get("api_key", "")
        ),
    )
    _set_runtime_config(
        "chatterbox",
        "model_id",
        (
            st.session_state.get(
                "chatterbox_model_input",
                config.chatterbox.get("model_id") or DEFAULT_CHATTERBOX_MODEL,
            )
            or DEFAULT_CHATTERBOX_MODEL
        ).strip(),
    )
    _set_runtime_config(
        "chatterbox",
        "voices",
        _parse_chatterbox_voices(
            st.session_state.get(
                "chatterbox_voices_input",
                config.chatterbox.get("voices") or DEFAULT_CHATTERBOX_VOICES,
            )
        ),
    )


def _sync_kokoro_config_from_session_state():
    # Voice catalog renders before setting input fields; sync browser state first to ensure this rerun
    # uses the new endpoint and manual voice configurations without requiring another widget interaction.
    _set_runtime_config(
        "kokoro",
        "base_url",
        (
            st.session_state.get(
                "kokoro_base_url_input",
                config.kokoro.get("base_url") or DEFAULT_KOKORO_BASE_URL,
            )
            or ""
        ).strip(),
    )
    _set_runtime_config(
        "kokoro",
        "api_key",
        st.session_state.get(
            "kokoro_api_key_input", config.kokoro.get("api_key", "")
        ),
    )
    _set_runtime_config(
        "kokoro",
        "model_id",
        (
            st.session_state.get(
                "kokoro_model_input",
                config.kokoro.get("model_id") or DEFAULT_KOKORO_MODEL,
            )
            or DEFAULT_KOKORO_MODEL
        ).strip(),
    )
    _set_runtime_config(
        "kokoro",
        "voices",
        _parse_chatterbox_voices(
            st.session_state.get(
                "kokoro_voices_input",
                config.kokoro.get("voices") or DEFAULT_KOKORO_VOICES,
            )
        ),
    )


def _get_kokoro_voice_options(saved_voice_name: str) -> list[str]:
    """Short-cache remote voice catalog in session; preserve previous selection on disconnection without treating outages as voice changes."""
    if config.kokoro.get("voices"):
        return voice.get_kokoro_voices()

    # Retain only one cache entry for the current service. Re-query immediately when endpoint/credentials change,
    # without caching plaintext keys; avoid blocking other UI interactions for 5 seconds on a known offline service within 30s.
    signature = (
        (config.kokoro.get("base_url") or "").strip().rstrip("/"),
        _credential_signature(config.kokoro.get("api_key", "")),
    )
    catalog = st.session_state.get("kokoro_voice_catalog", {})
    if catalog.get("signature") != signature:
        catalog = {"signature": signature, "voices": [], "checked_at": None}
    now = time.monotonic()
    if catalog["checked_at"] is None or now - catalog["checked_at"] >= 30:
        fetched = voice.get_kokoro_voices(fallback=False)
        catalog.update(checked_at=now, available=bool(fetched))
        if fetched:
            catalog["voices"] = fetched
        st.session_state["kokoro_voice_catalog"] = catalog

    options = list(catalog["voices"])
    if not catalog["available"]:
        st.warning(tr("Kokoro Voices Unavailable"))
        # Cache may be empty on initial launch; retain the configured selection. Only a successfully returned
        # new catalog after reconnection can determine whether an old voice was actually deleted by the server.
        if voice.is_kokoro_voice(saved_voice_name) and saved_voice_name not in options:
            options.insert(0, saved_voice_name)
    return options or [f"kokoro:{voice.KOKORO_DEFAULT_VOICE}"]


def _detect_audio_mime(audio_file: str, audio_bytes: bytes) -> str:
    # Some OpenAI-compatible TTS services (e.g. travisvn/chatterbox-tts-api) return WAV content even when
    # requesting response_format=mp3. If WebUI audio preview fixedly uses audio/mp3, browsers may fail to play it,
    # so detect the true format by file header here.
    header = audio_bytes[:12]
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "audio/wav"
    if header.startswith(b"ID3") or header[:2] in (
        b"\xff\xfb",
        b"\xff\xf3",
        b"\xff\xf2",
    ):
        return "audio/mp3"
    if header.startswith(b"OggS"):
        return "audio/ogg"
    ext = os.path.splitext(audio_file)[1].lower()
    return {
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }.get(ext, "audio/mp3")


def _build_uploaded_file_path(uploaded_file, target_dir, allowed_extensions, prefix):
    """Generate a controlled server-side storage path for browser-uploaded files."""
    original_name = os.path.basename(str(uploaded_file.name or ""))
    extension = os.path.splitext(original_name)[1].lower()
    if extension not in allowed_extensions:
        logger.warning(
            f"reject unsupported uploaded file extension: {original_name or '<empty>'}"
        )
        raise ValueError("unsupported uploaded file type")

    normalized_target_dir = os.path.realpath(target_dir)
    os.makedirs(normalized_target_dir, exist_ok=True)
    # Do not reuse browser-supplied filenames to avoid path separators, control characters, or accidental overwrites.
    # UUID is used solely for server persistence and does not alter the original filename displayed in the upload widget.
    file_path = os.path.realpath(
        os.path.join(normalized_target_dir, f"{prefix}-{uuid4().hex}{extension}")
    )
    if os.path.commonpath([normalized_target_dir, file_path]) != normalized_target_dir:
        logger.warning(f"invalid uploaded file path: {file_path}")
        raise ValueError("invalid uploaded file path")
    return file_path


def _save_uploaded_local_materials(uploaded_files):
    """Validate a WebUI material batch and undo earlier files on failure."""
    local_videos_dir = utils.storage_dir("local_videos", create=True)
    materials = []
    persisted = []
    saved_paths = []
    try:
        for uploaded_file in uploaded_files:
            stored_name = material_upload_service.save_material_upload(
                uploaded_file.name, uploaded_file
            )
            file_path = os.path.join(local_videos_dir, stored_name)
            saved_paths.append(file_path)
            material_info = MaterialInfo()
            material_info.provider = "local"
            material_info.url = file_path
            materials.append(material_info)
            persisted.append(
                {
                    "provider": material_info.provider,
                    "url": material_info.url,
                    "duration": material_info.duration,
                }
            )
    except Exception:
        for file_path in saved_paths:
            try:
                os.remove(file_path)
            except OSError as exc:
                logger.warning(
                    f"failed to remove local material after batch error: "
                    f"path={file_path}, error={exc}"
                )
        raise
    return materials, persisted


def _initialize_session_state():
    """Centrally initialize page states preserved across reruns."""
    if not st.session_state.get("cross_post_recovery_checked"):
        # WebUI can run independently without FastAPI, so handle publishing states left from process restarts
        # during first session initialization. Do not write flag on recovery failure; subsequent reruns will retry.
        recovered = tm.recover_interrupted_cross_posts()
        if recovered is not None:
            st.session_state["cross_post_recovery_checked"] = True

    saved_ui_language = config.ui.get("language", "")
    browser_locale = st.context.locale
    initial_ui_language = utils.resolve_ui_language(
        saved_language=saved_ui_language,
        browser_locale=browser_locale,
        supported_languages=locales.keys(),
    )

    defaults = {
        "video_subject": "",
        "video_script": "",
        "video_terms": "",
        "paragraph_number_input": _saved_ui_number(
            "paragraph_number",
            1,
            llm.MIN_SCRIPT_PARAGRAPH_NUMBER,
            llm.MAX_SCRIPT_PARAGRAPH_NUMBER,
            int,
        ),
        "video_script_prompt": _saved_ui_text(
            "video_script_prompt",
            max_length=llm.MAX_SCRIPT_PROMPT_LENGTH,
        ),
        "custom_system_prompt": _saved_ui_text(
            "custom_system_prompt",
            llm.DEFAULT_SCRIPT_SYSTEM_PROMPT,
            llm.MAX_SCRIPT_SYSTEM_PROMPT_LENGTH,
        ),
        "match_materials_to_script": bool(
            config.app.get("match_materials_to_script", False)
        ),
        "custom_bgm_file_input": _saved_ui_text("custom_bgm_file"),
        "sonilo_bgm_prompt_input": _saved_ui_text(
            "sonilo_bgm_prompt",
            max_length=sonilo_service.MAX_PROMPT_LENGTH,
        ),
        "elevenlabs_music_prompt_input": _saved_ui_text(
            "elevenlabs_music_prompt",
            max_length=elevenlabs_music_service.MAX_PROMPT_LENGTH,
        ),
        "subtitle_enabled_checkbox": _saved_ui_bool("subtitle_enabled", True),
        "stroke_color_picker": _saved_ui_color("stroke_color", "#000000"),
        "stroke_width_slider": _saved_ui_number(
            "stroke_width", 1.5, 0.0, 10.0
        ),
        "loomloom_candidate_count": _saved_ui_number(
            "loomloom_candidate_count",
            3,
            1,
            loomloom.MAX_SCRIPT_CANDIDATES,
            int,
        ),
        "loomloom_script_duration_seconds": _saved_ui_number(
            "loomloom_script_duration_seconds", 60, 10, 600, int
        ),
        "ui_language": initial_ui_language,
        # Persisted local materials allow users to reuse them when modifying only script copy.
        "local_video_materials": [],
        # Generate button callback registers the task first so the top bar immediately shows running counts.
        "active_generation_tasks": {},
        # Most recent task submitted from the current page. After moving generation to background tasks,
        # the page fragment polls status using this ID; refreshes no longer depend on running legacy scripts.
        "current_generation_task_id": "",
        # LoomLoom quoting and execution must preserve identical inputs and clientRequestId across Streamlit reruns
        # to prevent network retries from creating duplicate paid tasks.
        "loomloom_script_batch": None,
        "loomloom_script_quote": None,
        "loomloom_script_input_signature": "",
        "loomloom_client_request_id": "",
        "loomloom_run_id": "",
        "loomloom_run_status": "",
        "loomloom_run_error": "",
        "loomloom_poll_failure_count": 0,
        "loomloom_poll_retry_after": 0.0,
        "loomloom_poll_paused": False,
        "loomloom_script_candidates": (),
        "loomloom_candidate_errors": (),
        "loomloom_selected_candidate": 0,
        "loomloom_video_batch": None,
        "loomloom_video_quote": None,
        "loomloom_video_input_signature": "",
        "loomloom_video_client_request_id": "",
        "loomloom_video_confirm_charge": False,
        "loomloom_video_quote_error_signature": "",
        "loomloom_video_quote_error": "",
        "loomloom_video_capability": None,
        "loomloom_video_capability_fingerprint": "",
        "loomloom_video_capability_load_attempt": "",
        "loomloom_video_capability_error": "",
        "loomloom_video_model_id": "",
        # When script or full voiceover is generated, consume this digest before creating the video count widget
        # to autofill recommended material counts; cleared after consumption to avoid overwriting manual adjustments.
        "loomloom_video_scene_autofill_digest": "",
        "wavespeed_confirm_charge": False,
        "volcengine_seedance_confirm_charge": False,
        "ofox_confirm_charge": False,
        "metaso_minimax_confirm_charge": False,
        "muapi_confirm_charge": False,
        # AI video charges per material segment; defaults to 1 segment, allowing users to increase counts after reviewing results.
        "loomloom_video_scene_count": _saved_ui_number(
            "loomloom_video_scene_count",
            1,
            1,
            loomloom.MAX_VIDEO_SCENES,
            int,
        ),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_initialize_session_state()


def tr(key):
    loc = locales.get(st.session_state["ui_language"], {})
    value = loc.get("Translation", {}).get(key)
    if value is not None:
        return value
    # Prioritize Chinese and English for new features. When other languages lack a translation, fall back to English
    # to avoid losing sync across multiple locales from copying English text; display the raw key only if English lacks it too.
    return locales.get("en", {}).get("Translation", {}).get(key, key)


# -----------------------------------------------------------------------------
# Task management: history scanning, running status, parameter restoration, and table interactions
# -----------------------------------------------------------------------------


def _format_task_time(timestamp):
    if not timestamp:
        return "-"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _format_task_subject(subject, max_length=30):
    subject = str(subject or "").replace("\n", " ").strip()
    if len(subject) <= max_length:
        return subject or "-"
    return f"{subject[:max_length]}..."


def _safe_load_task_script(task_path):
    script_file = os.path.join(task_path, "script.json")
    if not os.path.isfile(script_file):
        return {}

    try:
        with open(script_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            logger.warning(f"task script data is not an object: {script_file}")
            return {}
        return payload
    except Exception as e:
        logger.warning(f"failed to read task script data: {script_file}, {e}")
        return {}


def _find_final_task_video(task_path: str) -> str:
    """
    Return the final video with the lowest index in the task directory.

    The synthesis pipeline also produces combined, temp-clip, and MoviePy temporary files;
    since these do not indicate successful completion, only ``final-<index>.<ext>`` is accepted here.
    """
    try:
        files = os.listdir(task_path)
    except OSError:
        return ""

    candidates = []
    for file_name in files:
        match = _FINAL_VIDEO_PATTERN.fullmatch(file_name)
        if match and os.path.isfile(os.path.join(task_path, file_name)):
            candidates.append((int(match.group("index")), file_name))

    if not candidates:
        return ""

    _, file_name = min(candidates, key=lambda item: item[0])
    return os.path.join(task_path, file_name)


def _build_restore_upload_requirements(params: Mapping) -> dict:
    """
    Record uploaded file dependencies from historical tasks that Streamlit cannot automatically restore.

    Browsers disallow programmatic repopulation of file_uploader, so local materials and custom audio dependencies
    must be tracked separately to check if they have been replenished or replaced before re-generating.
    """
    return {
        "local_materials": params.get("video_source") == "local",
        "custom_audio": bool(params.get("custom_audio_file")),
        "original_voice_name": params.get("voice_name") or "",
    }


def _get_unmet_restore_upload_requirements(
    requirements: Mapping | None,
    *,
    video_source: str,
    voice_name: str,
    has_local_materials: bool,
    has_custom_audio: bool,
    voice_mode: str | None = None,
) -> set[str]:
    """Return unmet historical uploaded file dependencies for the current form."""
    requirements = requirements or {}
    unmet = set()

    if (
        requirements.get("local_materials")
        and video_source == "local"
        and not has_local_materials
    ):
        unmet.add("local_materials")

    if requirements.get("custom_audio") and not has_custom_audio:
        if voice_mode is not None:
            # The new WebUI uses explicit voiceover modes. Switching to auto or no voiceover indicates
            # intentional replacement of historical uploaded audio; re-upload is only required when staying in upload mode.
            if voice_mode == VOICE_MODE_UPLOAD:
                unmet.add("custom_audio")
        elif voice_name == requirements.get("original_voice_name", ""):
            # Preserve compatibility with legacy callers determining mode by voice name, avoiding breakage for APIs and test tools.
            unmet.add("custom_audio")

    return unmet


def _queue_task_restore(task_id):
    # The task list runs inside a fragment and cannot directly modify main form widget states that have already been created.
    # Record candidate tasks and trigger a full-page rerun here; confirmation and parameter restoration are handled by the main page.
    st.session_state["task_restore_candidate_id"] = task_id
    st.session_state["task_manager_popover_nonce"] = (
        st.session_state.get("task_manager_popover_nonce", 0) + 1
    )
    st.rerun(scope="app")


def _normalize_task_state(state):
    if state in (
        const.TASK_STATE_COMPLETE,
        const.TASK_STATE_FAILED,
        const.TASK_STATE_PROCESSING,
    ):
        return state
    try:
        return int(state)
    except (TypeError, ValueError):
        return state


def _active_generation_tasks():
    tasks = st.session_state.setdefault("active_generation_tasks", {})
    if not isinstance(tasks, dict):
        tasks = {}
        st.session_state["active_generation_tasks"] = tasks
    return tasks


def _add_active_generation_task(task_id, subject=None):
    tasks = _active_generation_tasks()
    task = tasks.setdefault(task_id, {})
    task["subject"] = subject or task.get("subject") or task_id
    task["mtime"] = task.get("mtime") or datetime.now().timestamp()


def _remove_active_generation_task(task_id):
    tasks = _active_generation_tasks()
    if task_id in tasks:
        del tasks[task_id]
    if st.session_state.get("pending_generation_task_id") == task_id:
        del st.session_state["pending_generation_task_id"]


def _prepare_generation_task():
    # st.button on_click fires before the page script reruns. Pre-generating the task ID here allows
    # the top task management entry point to display the 'generating' count within the very same rerun.
    task_id = str(uuid4())
    st.session_state["pending_generation_task_id"] = task_id
    subject = st.session_state.get("video_subject") or st.session_state.get(
        "video_script"
    )
    _add_active_generation_task(task_id, subject=subject)


def _task_state_label(state, has_video):
    normalized_state = _normalize_task_state(state)
    if normalized_state == const.TASK_STATE_COMPLETE:
        return tr("Task Status Complete")
    if normalized_state == const.TASK_STATE_FAILED:
        return tr("Task Status Failed")
    if normalized_state == const.TASK_STATE_PROCESSING:
        return tr("Task Status Processing")
    if has_video:
        return tr("Task Status Complete")
    return tr("Task Status History")


def _task_state_filter_key(task):
    normalized_state = _normalize_task_state(task.get("state"))
    if normalized_state == const.TASK_STATE_PROCESSING:
        return "processing"
    if normalized_state == const.TASK_STATE_FAILED:
        return "failed"
    if normalized_state == const.TASK_STATE_COMPLETE or task["video_file"]:
        return "complete"
    return "history"


def _scan_history_tasks(limit=30):
    tasks_root = utils.task_dir()
    if not os.path.isdir(tasks_root):
        return []

    # Task management fragment refreshes every 2 seconds. Read lightweight directory metadata first and take recent tasks
    # before parsing script.json and video lists, avoiding repeated full scans when task history is extensive.
    task_entries = []
    try:
        with os.scandir(tasks_root) as entries:
            for entry in entries:
                try:
                    if entry.name.startswith(".") or not entry.is_dir(
                        follow_symlinks=False
                    ):
                        continue
                    task_entries.append(
                        (
                            entry.stat(follow_symlinks=False).st_mtime,
                            entry.name,
                            entry.path,
                        )
                    )
                except OSError as e:
                    # A single task directory might be in the middle of deletion; do not fail the entire task panel because of it.
                    logger.debug(f"skip unavailable task directory: {entry.path}, {e}")
    except OSError as e:
        logger.warning(f"failed to scan task directory: {tasks_root}, {e}")
        return []

    task_entries.sort(key=lambda item: item[0], reverse=True)
    tasks = []
    for mtime, name, task_path in task_entries[:limit]:
        script_data = _safe_load_task_script(task_path)
        params_data = script_data.get("params", {})
        if not isinstance(params_data, dict):
            params_data = {}
        script_text = script_data.get("script", "")
        if not isinstance(script_text, str):
            script_text = ""
        video_file = _find_final_task_video(task_path)
        subject = (
            params_data.get("video_subject")
            or script_text[:40]
            or name
        )
        tasks.append(
            {
                "task_id": name,
                "subject": subject,
                "state": const.TASK_STATE_COMPLETE if video_file else None,
                "progress": 100 if video_file else 0,
                "mtime": mtime,
                "task_path": task_path,
                "video_file": video_file,
                "source": "history",
            }
        )

    return tasks


def _collect_task_summaries(limit=20):
    history_tasks = {task["task_id"]: task for task in _scan_history_tasks(limit=50)}
    active_tasks = _active_generation_tasks()

    try:
        runtime_tasks, _ = sm.state.get_all_tasks(1, 50)
    except Exception as e:
        logger.warning(f"failed to load runtime tasks: {e}")
        runtime_tasks = []

    # The paginated state view can omit this session's newer tasks after 50
    # older records. Read those active IDs directly so a completed or failed
    # task cannot remain labelled as processing forever.
    runtime_ids = {task.get("task_id") for task in runtime_tasks}
    for task_id in active_tasks:
        if task_id in runtime_ids:
            continue
        try:
            task = sm.state.get_task(task_id)
        except Exception as e:
            logger.warning(f"failed to load active task {task_id}: {e}")
            continue
        if task:
            runtime_tasks.append(task)

    for task in runtime_tasks:
        task_id = task.get("task_id", "")
        if not task_id:
            continue

        task_path = os.path.join(utils.task_dir(), task_id)
        history_task = history_tasks.get(task_id, {})
        video_files = task.get("videos") or []
        video_file = (
            video_files[0] if video_files else history_task.get("video_file", "")
        )
        subject = (
            task.get("video_subject")
            or history_task.get("subject")
            or (task.get("script", "")[:40] if task.get("script") else "")
            or task_id
        )
        task_mtime = active_tasks.get(task_id, {}).get("mtime") or history_task.get(
            "mtime", 0
        )
        if os.path.isdir(task_path):
            try:
                task_mtime = os.path.getmtime(task_path)
            except OSError:
                # Another session can delete this directory between isdir and
                # getmtime. Keep rendering the persisted task state.
                pass

        history_tasks[task_id] = {
            "task_id": task_id,
            "subject": subject,
            "state": task.get("state"),
            "cross_post_state": task.get("cross_post_state"),
            "progress": int(task.get("progress", 0) or 0),
            "mtime": task_mtime,
            "task_path": task_path,
            "video_file": video_file,
            "source": "runtime",
        }

    for task_id, active_task in active_tasks.items():
        history_task = history_tasks.get(task_id, {})
        if history_task and _task_state_filter_key(history_task) in {
            "complete",
            "failed",
        }:
            # The active flag in session only covers the brief window between submission and state storage registration.
            # Once background tasks end, prioritize the true final state so failed tasks are not shown as generating.
            continue

        task_path = os.path.join(utils.task_dir(), task_id)
        history_tasks[task_id] = {
            "task_id": task_id,
            "subject": active_task.get("subject")
            or history_task.get("subject")
            or task_id,
            "state": const.TASK_STATE_PROCESSING,
            "progress": history_task.get("progress", 0),
            "mtime": active_task.get("mtime")
            or history_task.get("mtime", datetime.now().timestamp()),
            "task_path": task_path,
            "video_file": history_task.get("video_file", ""),
            "source": "active",
        }

    tasks = list(history_tasks.values())
    return sorted(tasks, key=lambda item: item["mtime"], reverse=True)[:limit]


def _is_headless_server():
    # In Docker or headless server deployments, the WebUI process lacks access to the user's desktop environment:
    # xdg-open / webbrowser will silently fail inside the container. Fall back to in-browser video preview
    # and display the path instead of opening directories. macOS/Windows desktop environments remain unaffected.
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _open_task_path(task_path):
    tasks_root = os.path.abspath(utils.task_dir())
    normalized_path = os.path.abspath(task_path)
    if not normalized_path.startswith(tasks_root + os.sep):
        logger.warning(f"invalid task folder path: {normalized_path}")
        return
    if not os.path.isdir(normalized_path):
        return
    if _is_headless_server():
        # The storage directory is typically volume-mounted to the host; showing the relative path suffices to locate files.
        rel_path = os.path.relpath(normalized_path, os.path.dirname(tasks_root))
        st.toast(f"{tr('Open Task Folder')}: ./storage/{rel_path}", icon="📂")
        return
    webbrowser.open(f"file://{normalized_path}")


def _open_task_video(video_file):
    tasks_root = os.path.abspath(utils.task_dir())
    normalized_file = os.path.abspath(video_file)

    # Video paths originate from task directory scans or runtime state. Restrict opens strictly to files within the task directory
    # to prevent UI actions with abnormal paths from escalating into arbitrary local file openers.
    if not normalized_file.startswith(tasks_root + os.sep):
        logger.warning(f"invalid task video path: {normalized_file}")
        return
    if not os.path.isfile(normalized_file):
        logger.warning(f"task video does not exist: {normalized_file}")
        return

    if _is_headless_server():
        # In headless environments, render an embedded player inside the task panel instead of invoking system players.
        st.session_state["task_preview_video_file"] = normalized_file
        return

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", normalized_file])
        elif sys.platform.startswith("win"):
            os.startfile(normalized_file)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", normalized_file])
    except Exception as e:
        logger.error(f"failed to open task video: {normalized_file}, {e}")


def _delete_task(task_id, task_path, task_state=None):
    # The displayed status may lag behind background tasks. Check incoming status, current session active tasks,
    # and the latest status before deleting to avoid accidentally removing tasks that just started or produced intermediate videos.
    current_task = None
    try:
        current_task = sm.state.get_task(task_id)
    except Exception as e:
        logger.exception(f"failed to verify task state before deletion: {task_id}, {e}")
        return False

    task_snapshot = dict(current_task or {})
    task_snapshot.setdefault("state", task_state)
    if task_id in _active_generation_tasks():
        task_snapshot["state"] = const.TASK_STATE_PROCESSING

    if tm.is_task_busy(task_snapshot):
        logger.warning(f"refused to delete running task: {task_id}")
        return False

    tasks_root = os.path.abspath(utils.task_dir())
    normalized_path = os.path.abspath(task_path)

    # Deleting tasks removes task state and locally generated files. Confine deletions strictly to storage/tasks
    # to avoid abnormal task_path inputs deleting other local directories.
    if not normalized_path.startswith(tasks_root + os.sep):
        logger.warning(f"invalid task folder path for deletion: {normalized_path}")
        return False

    try:
        if os.path.isdir(normalized_path):
            shutil.rmtree(normalized_path)
        # Keep the persisted task available for retry when file removal fails
        # (for example, a video is still locked by another process on Windows).
        if hasattr(sm.state, "delete_task"):
            sm.state.delete_task(task_id)
        logger.info(f"deleted task: {task_id}")
        return True
    except Exception as e:
        logger.exception(f"failed to delete task: {task_id}, {e}")
        return False


def _count_processing_tasks(tasks):
    # The top task manager entry point only needs to display the number of 'generating' tasks.
    # Reuse the internal state key here to avoid multilingual display copy causing inconsistent statistics across languages.
    processing_task_ids = {
        task["task_id"]
        for task in tasks
        if _task_state_filter_key(task) == "processing"
    }
    return len(processing_task_ids)


def _task_manager_label(processing_count):
    label = tr("Task Manager")
    if processing_count <= 0:
        return label
    return f"{label} · {processing_count}"


def _build_video_download_name(subject, index, total):
    """Generate a cross-platform safe download filename based on the video topic."""
    safe_subject = _DOWNLOAD_FILENAME_INVALID_PATTERN.sub(" ", str(subject or ""))
    safe_subject = re.sub(r"\s+", " ", safe_subject).strip(" .")[:80].rstrip(" .")
    if not safe_subject:
        safe_subject = "video"
    # Win32 ignores trailing spaces and dots before extensions when matching device names.
    # Keep consistent with BGM upload rules to prevent names like ``CON .topic`` from bypassing reserved name protection.
    windows_basename = safe_subject.split(".", 1)[0].rstrip(" .").upper()
    if windows_basename in _WINDOWS_RESERVED_FILENAMES:
        safe_subject = f"_{safe_subject}"

    suffix = f"-{index}" if total > 1 else ""
    return f"{safe_subject}{suffix}.mp4"


def _render_task_table(filtered_tasks, key_prefix):
    with st.container(key=f"task_table_header_{key_prefix}"):
        header_cols = st.columns([1.1, 1.7, 3.0, 0.8, 1.6], vertical_alignment="center")
        header_cols[0].caption(tr("Task Status"))
        header_cols[1].caption(tr("Task Updated At"))
        header_cols[2].caption(tr("Task Subject"))
        header_cols[3].caption(tr("Task Progress"))
        header_cols[4].caption(tr("Task Actions"))

    if not filtered_tasks:
        st.info(tr("No Tasks Match Filter"))
        return

    visible_tasks = filtered_tasks[:12]
    list_height = min(390, max(96, len(visible_tasks) * 58))
    with st.container(height=list_height, border=False):
        for task in visible_tasks:
            task_id = task["task_id"]
            has_video = bool(task["video_file"] and os.path.isfile(task["video_file"]))
            is_processing = _task_state_filter_key(task) == "processing"
            is_busy = is_processing or tm.is_task_busy(task)
            has_restore_data = os.path.isfile(
                os.path.join(task["task_path"], "script.json")
            )
            safe_task_key = "".join(ch if ch.isalnum() else "_" for ch in task_id)[:40]

            # Use Streamlit native bordered containers + columns to retain row operations.
            # Compared to custom HTML/CSS tables, this is more resilient across Streamlit versions;
            # compared to dataframe, it preserves inline actions like play, open directory, and delete.
            with st.container(
                key=f"task_row_{key_prefix}_{safe_task_key}", border=True
            ):
                row_cols = st.columns(
                    [1.1, 1.7, 3.0, 0.8, 1.6],
                    vertical_alignment="center",
                )
                row_cols[0].write(_task_state_label(task["state"], has_video))
                row_cols[1].write(_format_task_time(task["mtime"]))
                row_cols[2].write(_format_task_subject(task["subject"]))
                row_cols[3].write(f"{task['progress']}%")

                action_cols = row_cols[4].columns(
                    4,
                    vertical_alignment="center",
                    gap="small",
                )
                with action_cols[0]:
                    play_label = tr("Play")
                    if st.button(
                        play_label,
                        key=f"play_task_{key_prefix}_{task_id}",
                        use_container_width=True,
                        icon=":material/play_arrow:",
                        help=play_label,
                        disabled=not has_video,
                    ):
                        _open_task_video(task["video_file"])

                with action_cols[1]:
                    open_label = tr("Open Task Folder")
                    if st.button(
                        open_label,
                        key=f"open_task_{key_prefix}_{task_id}",
                        use_container_width=True,
                        icon=":material/folder_open:",
                        help=open_label,
                    ):
                        _open_task_path(task["task_path"])

                with action_cols[2]:
                    restore_label = tr("Regenerate Task")
                    if st.button(
                        restore_label,
                        key=f"restore_task_{key_prefix}_{task_id}",
                        use_container_width=True,
                        icon=":material/replay:",
                        help=restore_label,
                        disabled=is_processing or not has_restore_data,
                    ):
                        _queue_task_restore(task_id)

                with action_cols[3]:
                    delete_label = tr("Delete Task")
                    delete_help = (
                        f"{delete_label} ({tr('Task Status Processing')})"
                        if is_busy
                        else delete_label
                    )
                    if st.button(
                        delete_label,
                        key=f"delete_task_{key_prefix}_{task_id}",
                        use_container_width=True,
                        icon=":material/delete:",
                        help=delete_help,
                        disabled=is_busy,
                    ):
                        if _delete_task(task_id, task["task_path"], task["state"]):
                            st.toast(tr("Task Deleted"))
                            st.rerun()
                        else:
                            st.error(tr("Task Delete Failed"))


def _render_task_manager_panel(tasks=None):
    tasks = tasks if tasks is not None else _collect_task_summaries()
    if not tasks:
        st.info(tr("No Tasks Yet"))
        return

    # Streamlit 1.59 supports lazy rendering of stateful Tabs. Reconstruct only the active list upon tab switching,
    # avoiding redundant recreation of 4 sets of task rows and buttons by the periodic fragment every 2 seconds.
    status_tabs = [
        ("all", tr("All Tasks")),
        ("processing", tr("Task Status Processing")),
        ("complete", tr("Task Status Complete")),
        ("failed", tr("Task Status Failed")),
    ]
    tabs = st.tabs(
        [label for _, label in status_tabs],
        key="task_manager_status_tabs",
        on_change="rerun",
    )
    for (status_key, _), tab in zip(status_tabs, tabs):
        if not tab.open:
            continue
        with tab:
            filtered_tasks = [
                task
                for task in tasks
                if status_key == "all" or _task_state_filter_key(task) == status_key
            ]
            _render_task_table(filtered_tasks, status_key)

    _render_task_video_preview()


def _render_task_video_preview():
    # In-browser fallback for 'Play' button in headless environments: render player at bottom of task panel.
    preview_file = st.session_state.get("task_preview_video_file")
    if not preview_file:
        return

    tasks_root = os.path.abspath(utils.task_dir())
    if not (
        preview_file.startswith(tasks_root + os.sep) and os.path.isfile(preview_file)
    ):
        st.session_state.pop("task_preview_video_file", None)
        return

    st.divider()
    preview_cols = st.columns([5, 1], vertical_alignment="center")
    task_name = os.path.basename(os.path.dirname(preview_file))
    preview_cols[0].caption(f"{os.path.basename(preview_file)} · {task_name}")
    closed = preview_cols[1].button(
        "✕",
        key="close_task_video_preview",
        use_container_width=True,
        help=tr("Close"),
    )
    if closed:
        st.session_state.pop("task_preview_video_file", None)
        return
    st.video(preview_file)


@st.fragment(run_every="2s")
def _render_task_manager_entry():
    # Tasks may be triggered by current or other pages. The entry point refreshes periodically via a separate fragment,
    # updating only task counts and popover contents without interrupting main page form inputs.
    task_summaries = _collect_task_summaries()
    processing_task_count = _count_processing_tasks(task_summaries)
    with st.container(key="task_manager_entry", width="content"):
        with st.popover(
            _task_manager_label(processing_task_count),
            width="content",
            key=(
                "task_manager_popover_"
                f"{st.session_state.get('task_manager_popover_nonce', 0)}"
            ),
        ):
            _render_task_manager_panel(task_summaries)


def _load_task_restore_payload(task_id):
    tasks_root = os.path.realpath(utils.task_dir())
    task_path = os.path.realpath(os.path.join(tasks_root, str(task_id)))
    try:
        if os.path.commonpath([tasks_root, task_path]) != tasks_root:
            raise ValueError("task path is outside the task directory")
    except ValueError as e:
        logger.warning(f"invalid task restore path: {task_id}, {e}")
        return None

    script_data = _safe_load_task_script(task_path)
    raw_params = script_data.get("params")
    if not isinstance(raw_params, dict):
        logger.warning(f"task has no restorable parameters: {task_id}")
        return None

    params_input = dict(raw_params)
    if script_data.get("script"):
        params_input["video_script"] = script_data["script"]
    if script_data.get("search_terms"):
        params_input["video_terms"] = script_data["search_terms"]

    try:
        params = VideoParams.model_validate(params_input).model_dump(mode="json")
    except Exception as e:
        logger.warning(f"failed to validate task restore parameters: {task_id}, {e}")
        return None

    return {
        "task_id": str(task_id),
        "subject": params.get("video_subject") or script_data.get("script") or task_id,
        "params": params,
    }


def _infer_tts_server_from_voice(voice_name):
    if voice.is_no_voice(voice_name):
        return voice.NO_VOICE_NAME
    if voice.is_siliconflow_voice(voice_name):
        return "siliconflow"
    if voice.is_gemini_voice(voice_name):
        return "gemini-tts"
    if voice.is_mimo_voice(voice_name):
        return "mimo-tts"
    if voice.is_minimax_voice(voice_name):
        return "minimax-tts"
    if voice.is_elevenlabs_voice(voice_name):
        return "elevenlabs"
    if voice.is_chatterbox_voice(voice_name):
        return "chatterbox"
    if voice.is_kokoro_voice(voice_name):
        return "kokoro"
    if voice.is_fish_audio_voice(voice_name):
        return "fish_audio"
    if voice.is_voxcpm_voice(voice_name):
        return "voxcpm"
    if voice.is_azure_v2_voice(voice_name):
        return "azure-tts-v2"
    return "azure-tts-v1"


def _set_stable_widget_value(key, value):
    if value is not None:
        st.session_state[localized_widget_key(key)] = value


def _apply_pending_task_restore():
    payload = st.session_state.pop("task_restore_payload", None)
    if not payload:
        return False

    _apply_restored_params(payload["params"])
    st.session_state["task_restore_succeeded"] = True
    logger.info(f"restored task configuration: {payload['task_id']}")
    return True


def _apply_restored_params(params):
    """
    Write a complete set of generation parameters back to page widget states.

    Historical task restoration and settings preset import share the same parameter model and implementation,
    preventing drift when new fields are added. Callers must execute this before rendering any widgets;
    otherwise, Streamlit disallows modifying already-instantiated widget state.
    """
    video_terms = params.get("video_terms") or ""
    if isinstance(video_terms, list):
        video_terms = ", ".join(str(term) for term in video_terms)

    # Script copy and advanced script settings.
    st.session_state["video_subject"] = params.get("video_subject") or ""
    st.session_state["video_script"] = params.get("video_script") or ""
    st.session_state["video_terms"] = str(video_terms)
    _set_stable_widget_value(
        "script_language_select", params.get("video_language") or ""
    )
    st.session_state["paragraph_number_input"] = params.get("paragraph_number", 1)
    st.session_state["video_script_prompt"] = params.get("video_script_prompt") or ""
    st.session_state["custom_system_prompt"] = (
        params.get("custom_system_prompt") or llm.DEFAULT_SCRIPT_SYSTEM_PROMPT
    )

    # Video settings. Material upload widgets cannot be populated by the server, so local materials require user reselection.
    video_source = params.get("video_source") or "pexels"
    _set_stable_widget_value("video_source_select", video_source)
    _set_stable_widget_value(
        "video_concat_mode_select", params.get("video_concat_mode") or "random"
    )
    _set_stable_widget_value(
        "video_transition_mode_select",
        params.get("video_transition_mode") or VideoTransitionMode.none.value,
    )
    _set_stable_widget_value(
        f"video_aspect_for_{video_source}",
        params.get("video_aspect") or VideoAspect.portrait.value,
    )
    _set_stable_widget_value(
        "video_fit_mode_select",
        params.get("video_fit_mode") or VideoFitMode.cover.value,
    )
    _set_stable_widget_value(
        "image_motion_select", params.get("image_motion") or ImageMotion.zoom.value
    )
    _set_stable_widget_value(
        "video_clip_duration_select", params.get("video_clip_duration", 3)
    )
    _set_stable_widget_value(
        "video_clip_speed_slider",
        # APIs may accept speeds beyond WebUI bounds; generation safely normalizes them, but historical tasks
        # may retain raw values. Re-normalize before restoring to avoid injecting out-of-bounds, NaN, or inf values into sliders.
        utils.normalize_clip_speed(params.get("video_clip_speed", 1.0)),
    )
    _set_stable_widget_value("video_count_select", params.get("video_count", 1))
    st.session_state["match_materials_to_script"] = bool(
        params.get("match_materials_to_script", False)
    )

    # Audio settings. Infer TTS server from historical voice_name if missing in legacy tasks.
    voice_name = params.get("voice_name") or voice.NO_VOICE_NAME
    tts_server = _infer_tts_server_from_voice(voice_name)
    if params.get("custom_audio_file"):
        voice_mode = VOICE_MODE_UPLOAD
    elif voice.is_no_voice(voice_name):
        voice_mode = VOICE_MODE_NONE
    else:
        voice_mode = VOICE_MODE_TTS
    _set_stable_widget_value("voice_mode_control", voice_mode)
    if tts_server != voice.NO_VOICE_NAME:
        _set_stable_widget_value("tts_server_select", tts_server)
        _set_stable_widget_value(f"speech_synthesis_select_{tts_server}", voice_name)
    _set_stable_widget_value("voice_volume_select", params.get("voice_volume", 1.0))
    _set_stable_widget_value("voice_rate_select", params.get("voice_rate", 1.0))
    bgm_type = params.get("bgm_type") or ""
    _set_stable_widget_value("bgm_type_select", bgm_type)
    _set_stable_widget_value("bgm_volume_select", params.get("bgm_volume", 0.2))
    if bgm_type == "preset" and params.get("bgm_file"):
        # The preset song widget uses filename as a stable business value. Historical tasks may store absolute or
        # relative paths; normalizing to basename matches safely enumerated song lists.
        _set_stable_widget_value(
            "preset_song_select", os.path.basename(str(params["bgm_file"]))
        )
    st.session_state["custom_bgm_file_input"] = params.get("bgm_file") or ""
    st.session_state["sonilo_bgm_prompt_input"] = (
        params.get("video_music_prompt") or params.get("sonilo_bgm_prompt") or ""
    )
    st.session_state["elevenlabs_music_prompt_input"] = (
        params.get("video_music_prompt") or ""
    )

    # Subtitle settings. Clamp out-of-bounds numbers from legacy tasks so sliders can initialize cleanly.
    st.session_state["subtitle_enabled_checkbox"] = bool(
        params.get("subtitle_enabled", True)
    )
    _set_stable_widget_value("font_name_select", params.get("font_name") or "")
    _set_stable_widget_value(
        "subtitle_position_select", params.get("subtitle_position") or "bottom"
    )
    _set_stable_widget_value(
        "subtitle_display_mode_select", params.get("subtitle_display_mode") or "sentence"
    )
    _set_stable_widget_value(
        "subtitle_animation_select", params.get("subtitle_animation") or "none"
    )
    custom_position = min(100.0, max(0.0, float(params.get("custom_position", 70.0))))
    st.session_state["custom_position_input"] = str(custom_position)
    st.session_state["font_color_picker"] = params.get("text_fore_color") or "#FFFFFF"
    st.session_state["font_size_slider"] = min(
        100, max(30, int(params.get("font_size", 60)))
    )
    st.session_state["stroke_color_picker"] = params.get("stroke_color") or "#000000"
    st.session_state["stroke_width_slider"] = min(
        10.0, max(0.0, float(params.get("stroke_width", 1.5)))
    )
    background_color = params.get("text_background_color")
    background_enabled = bool(background_color)
    st.session_state["subtitle_background_enabled_checkbox"] = background_enabled
    if isinstance(background_color, str):
        st.session_state["subtitle_background_color_picker"] = background_color
    st.session_state["rounded_subtitle_background_checkbox"] = bool(
        params.get("rounded_subtitle_background", False) and background_enabled
    )

    st.session_state.pop("local_video_materials_uploader", None)
    # Historical tasks only store material paths without guaranteeing file existence in the current environment.
    # Clear cached uploaded materials in current page to prevent accidental reuse across tasks.
    st.session_state["local_video_materials"] = []
    st.session_state.pop("custom_audio_file_uploader", None)
    st.session_state.pop("voxcpm_reference_audio_uploader", None)
    st.session_state.pop(VOXCPM_REFERENCE_AUDIO_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY, None)
    st.session_state.pop("voxcpm_prompt_audio_uploader", None)
    st.session_state.pop(VOXCPM_PROMPT_AUDIO_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_PROMPT_TEXT_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_HIGH_FIDELITY_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_PROMPT_EXAMPLE_MODE_SESSION_KEY, None)
    st.session_state.pop("custom_bgm_uploader", None)
    st.session_state.pop("custom_bgm_validation", None)
    st.session_state["task_restore_upload_requirements"] = (
        _build_restore_upload_requirements(params)
    )

    return True


def _dismiss_task_restore_dialog():
    st.session_state.pop("task_restore_candidate_id", None)


@st.dialog(
    tr("Regenerate Task"),
    width="small",
    on_dismiss=_dismiss_task_restore_dialog,
)
def _render_task_restore_dialog(task_id):
    payload = _load_task_restore_payload(task_id)
    if payload is None:
        st.error(tr("Task Restore Failed"))
        if st.button(tr("Cancel"), key="cancel_invalid_task_restore"):
            st.session_state.pop("task_restore_candidate_id", None)
            st.rerun(scope="app")
        return

    st.write(tr("Regenerate Task Confirmation"))
    st.caption(_format_task_subject(payload["subject"], max_length=80))
    cancel_col, load_col = st.columns(2)
    if cancel_col.button(
        tr("Cancel"),
        key="cancel_task_restore",
        use_container_width=True,
    ):
        st.session_state.pop("task_restore_candidate_id", None)
        st.rerun(scope="app")
    if load_col.button(
        tr("Load Task Configuration"),
        key="confirm_task_restore",
        type="primary",
        use_container_width=True,
    ):
        st.session_state["task_restore_payload"] = payload
        st.session_state.pop("task_restore_candidate_id", None)
        st.rerun(scope="app")


def _dismiss_settings_dialog():
    """Dismiss the settings dialog and ensure the next full-page rerun does not automatically reopen it."""
    st.session_state["settings_dialog_open"] = False


def _open_settings_dialog(target_tab=None):
    """Open the settings dialog, optionally navigating directly to a specific tab."""
    st.session_state["settings_dialog_open"] = True
    if target_tab:
        # Save stable business IDs rather than translated labels; resolve labels based on current locale
        # right before creating tabs to prevent old strings becoming invalid options upon language switch.
        st.session_state["settings_dialog_target_tab"] = target_tab


def _open_material_settings_dialog():
    """Callback for video source components: navigate directly to material service settings."""
    _open_settings_dialog("material")


def _render_brand(available_update: str | None = None):
    """Render project branding, current version, and optional update link."""
    update_link = ""
    if available_update:
        update_label = html.escape(
            tr("Update Available").format(version=available_update)
        )
        # Streamlit parses injected HTML with Markdown. Keep links on a single line to prevent indentation
        # from being treated as code blocks, which would render raw HTML markup directly.
        update_link = (
            '<a class="mpt-brand__update" '
            f'href="{version_checker.LATEST_RELEASE_PAGE_URL}" '
            'target="_blank" rel="noopener noreferrer" '
            f'aria-label="{update_label}" title="{update_label}">'
            f"{update_label}</a>"
        )
    st.markdown(
        f"""
        <h1 class="mpt-brand">
            <span class="mpt-brand__name">MoneyPrinterTurbo</span>
            <a class="mpt-brand__version"
               href="https://github.com/harry0703/MoneyPrinterTurbo"
               target="_blank"
               rel="noopener noreferrer"
               aria-label="Open MoneyPrinterTurbo on GitHub"
               title="Open project on GitHub">v{html.escape(str(config.project_version))}</a>
            {update_link}
        </h1>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every="1s")
def _render_pending_version_check():
    """Refresh only the brand section while checking, avoiding blocking or repeatedly executing the full form."""
    snapshot = version_checker.poll_available_update(config.project_version)
    if snapshot.complete:
        # Perform a single full-page rerun once the check completes, converting the top bar to static rendering
        # and halting fragment polling. Occurs after the background request completes without delaying initial page content.
        st.rerun(scope="app")
    _render_brand()


def _render_top_bar():
    """Render the top bar consisting of branding, task manager, settings, and language switcher."""
    # The top bar is divided into brand and action sections. On narrow viewports, Streamlit wraps the two sections,
    # and action controls wrap internally based on available width.
    with st.container(key="top_bar"):
        brand_col, actions_col = st.columns(
            [3.5, 2.0],
            vertical_alignment="center",
            gap="small",
        )

    with brand_col:
        update_snapshot = version_checker.poll_available_update(config.project_version)
        if update_snapshot.complete:
            _render_brand(update_snapshot.available_version)
        else:
            _render_pending_version_check()

    with actions_col:
        with st.container(
            key="top_bar_actions",
            horizontal=True,
            horizontal_alignment="right",
            vertical_alignment="center",
            gap="small",
            width="stretch",
        ):
            _render_task_manager_entry()

            st.button(
                tr("Settings"),
                key="open_settings_dialog_button",
                type="secondary",
                icon=":material/settings:",
                width="content",
                on_click=_open_settings_dialog,
            )

            language_codes = list(locales.keys())
            selected_index = 0
            for i, code in enumerate(language_codes):
                if code == st.session_state.get("ui_language", ""):
                    selected_index = i

            selected_language_code = st.selectbox(
                "Language / 语言",
                options=language_codes,
                index=selected_index,
                format_func=lambda code: locales[code].get("Language", code),
                key="top_language_code_selector",
                label_visibility="collapsed",
                width=180,
            )
            if selected_language_code:
                previous_language = st.session_state.get("ui_language", "")
                if selected_language_code != previous_language:
                    logger.info(
                        "UI language changed by user: "
                        f"previous_language={previous_language or '<empty>'}, "
                        f"selected_language={selected_language_code}"
                    )
                    st.session_state["ui_language"] = selected_language_code
                    # Browser auto-detection only affects the current session; write to config.toml only when the user
                    # explicitly switches the selectbox, giving priority to this explicit choice in future sessions.
                    _set_runtime_config("ui", "language", selected_language_code)
                    _save_runtime_config()
                    # Switching language reruns the top bar first, triggering a rerun before body widgets render.
                    # Explicitly preserve draft contents to prevent Streamlit resetting input topic, script, and keywords
                    # when cleaning up old widget states for the new language page.
                    for content_key in ("video_subject", "video_script", "video_terms"):
                        if content_key in st.session_state:
                            st.session_state[content_key] = st.session_state[content_key]
                    # Force rerun after language change to prevent the selectbox from showing stale language text.
                    st.rerun()


support_locales = [
    "ca-ES",
    "zh-CN",
    "zh-HK",
    "zh-TW",
    "de-DE",
    "en-US",
    "es-ES",
    "fr-FR",
    "it-IT",
    "ru-RU",
    "vi-VN",
    "th-TH",
    "tr-TR",
]


# -----------------------------------------------------------------------------
# Common UI components, resource caching, and logging
# -----------------------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def get_all_fonts():
    # Font directories rarely change, but Streamlit reruns the page on every widget interaction. Short-lived caching
    # avoids repeated os.walk calls while ensuring newly added fonts are discovered within 30 seconds.
    fonts = []
    for root, dirs, files in os.walk(font_dir):
        for file in files:
            if file.endswith(".ttf") or file.endswith(".ttc"):
                # Keep each option tied to its file, including nested fonts.
                relative_path = os.path.relpath(os.path.join(root, file), font_dir)
                fonts.append(relative_path.replace(os.sep, "/"))
    fonts.sort()
    return fonts


@st.cache_data(ttl=30, show_spinner=False)
def get_all_songs():
    # BGM shares the same short-lived caching strategy as fonts without permanent caching, balancing rerun performance
    # with manual addition of music files during runtime.
    songs = []
    for root, dirs, files in os.walk(song_dir):
        for file in files:
            if file.endswith(".mp3"):
                songs.append(file)
    return songs


def open_task_folder(task_id):
    try:
        # task_id should always be a server-generated UUID. Validate its format first to prevent abnormal values
        # escaping the task directory via path traversal, and avoid shell interpolation of special characters when opening directories.
        normalized_task_id = str(UUID(str(task_id)))
        tasks_root = os.path.abspath(utils.task_dir())
        path = os.path.abspath(os.path.join(tasks_root, normalized_task_id))

        # Even if UUID validation passes, reconfirm the resolved path remains inside the task root
        # to prevent path traversal risks if future callers modify the source of task_id.
        if not path.startswith(tasks_root + os.sep):
            logger.warning(f"invalid task folder path: {path}")
            return

        if os.path.isdir(path):
            webbrowser.open(f"file://{path}")
    except Exception as e:
        logger.exception(f"failed to open task folder: task_id={task_id}, error={e}")


@st.cache_resource
def init_log():
    # The base logging handler is a process-level resource, not page session state. Streamlit reruns
    # the page script on every widget interaction, and hot-reloading can invalidate caches. Logging initialization
    # must accurately replace the terminal handler only, without clearing temporary WebUI handlers used by running tasks.
    _lvl = get_logging_options()[0]

    return configure_terminal_logger(
        sys.stdout,
        level=_lvl,
        colorize=True,
    )


init_log()


def tr_optional(key, fallback_language=""):
    loc = locales.get(st.session_state["ui_language"], {})
    value = loc.get("Translation", {}).get(key, "")
    if not value and fallback_language:
        fallback_loc = locales.get(fallback_language, {})
        value = fallback_loc.get("Translation", {}).get(key, "")
    return value if value else ""


def render_onboarding_tour():
    # The tour only covers three stable entry points without attempting to control Dialogs, Tabs, or business forms.
    # This helps new users understand the end-to-end flow without coupling tour state to Streamlit's dynamic component lifecycles.
    steps = [
        Tour.bind(
            "open_settings_dialog_button",
            title=tr("Onboarding Model Settings Title"),
            desc=tr("Onboarding Model Settings Description"),
            side="bottom",
            align="end",
        ),
        Tour.bind(
            "main_settings_grid",
            title=tr("Onboarding Creation Settings Title"),
            desc=tr("Onboarding Creation Settings Description"),
            side="top",
            align="center",
        ),
        Tour.bind(
            "generate_video_button",
            title=tr("Onboarding Generate Video Title"),
            desc=tr("Onboarding Generate Video Description"),
            side="top",
            align="center",
        ),
    ]

    # streamlit-tour 1.1.0 does not expose navigation labels in its Python constructor parameters, but underlying
    # Driver.js supports overriding button text in each step's popover config. Inject localized copy here and HTML-escape
    # contents since the component renders these fields via innerHTML.
    previous_text = html.escape(tr("Onboarding Previous"))
    next_text = html.escape(tr("Onboarding Next"))
    done_text = html.escape(tr("Onboarding Done"))
    for index, step in enumerate(steps):
        step.popover["prevBtnText"] = f"&larr; {previous_text}"
        # Driver.js overrides variable-interpolated progress templates when merging single-step configs, so write
        # the current step and total steps directly to avoid rendering unparsed {{current}} placeholders.
        step.popover["progressText"] = f"{index + 1} / {len(steps)}"
        if index == len(steps) - 1:
            step.popover["doneBtnText"] = done_text
        else:
            step.popover["nextBtnText"] = f"{next_text} &rarr;"

    tour = Tour(
        steps=steps,
        key=ONBOARDING_TOUR_KEY,
        show_progress=True,
        animate=True,
        overlay_opacity=0.55,
        one_time_tour=True,
    )

    # Trigger only once per Streamlit session. Completion status is determined by the component via browser
    # localStorage to prevent the tour from popping up repeatedly across page reruns or regular widget interactions.
    auto_start_key = f"{ONBOARDING_TOUR_KEY}-auto-started"
    if not st.session_state.get(auto_start_key, False):
        st.session_state[auto_start_key] = True
        tour.start()


def _render_generation_logs(task_id):
    """Render background task log snapshots without accessing Streamlit session state from worker threads."""
    if config.ui.get("hide_log", False):
        return

    log_records = webui_task.get_task_logs(task_id)
    if not log_records:
        return

    st.code("\n".join(log_records))


def _render_generation_task_snapshot(task_id, task):
    """Render progress, failure reasons, or final video outputs based on snapshots in state storage."""
    if not task:
        st.info(tr("Generating Video"))
        _render_generation_logs(task_id)
        return

    state = _normalize_task_state(task.get("state"))
    progress = max(0, min(100, int(task.get("progress", 0) or 0)))
    if state == const.TASK_STATE_PROCESSING:
        st.info(tr("Generating Video"))
        st.progress(
            progress,
            text=f"{tr('Task Progress')}: {progress}%",
        )
        _render_generation_logs(task_id)
        return

    video_files = task.get("videos") or []
    if state == const.TASK_STATE_FAILED:
        error = str(task.get("error") or "").strip()
        message = tr("Video Generation Failed")
        st.error(f"{message}: {error}" if error else message)
        if not video_files:
            _render_generation_logs(task_id)
            return
    elif state != const.TASK_STATE_COMPLETE or not video_files:
        st.error(tr("Video Generation Failed"))
        _render_generation_logs(task_id)
        return

    if state == const.TASK_STATE_COMPLETE:
        st.success(tr("Video Generation Completed"))
    for warning in task.get("warnings") or []:
        if isinstance(warning, Mapping) and warning.get("code") == "batch_materials_reused":
            st.warning(
                tr("Batch Material Reuse Warning").format(
                    index=warning.get("video_index", ""), count=warning.get("count", 0)
                )
            )
        elif isinstance(warning, Mapping) and warning.get("code") == "sonilo_bgm_failed":
            st.warning(
                tr("Sonilo BGM Fallback Warning").format(
                    index=warning.get("video_index", "")
                )
            )
        elif (
            isinstance(warning, Mapping)
            and warning.get("code") == "elevenlabs_bgm_failed"
        ):
            st.warning(
                tr("ElevenLabs BGM Fallback Warning").format(
                    index=warning.get("video_index", "")
                )
            )
        else:
            st.warning(str(warning))

    available_videos = [
        (index, url)
        for index, url in enumerate(video_files)
        if os.path.isfile(url)
    ]
    for index, url in enumerate(video_files):
        if os.path.isfile(url):
            continue
        logger.warning(
            f"generated video is unavailable: "
            f"task_id={task_id}, video_file={url}"
        )

    try:
        if not available_videos:
            st.warning(tr("Generated Video Files Unavailable"))
        else:
            player_cols = st.columns(len(available_videos) * 2 + 1)
            for player_index, (video_index, url) in enumerate(available_videos):
                with player_cols[player_index * 2 + 1]:
                    st.video(url)
                    download_label = tr("Download Video")
                    if len(video_files) > 1:
                        download_label = f"{download_label} {video_index + 1}"
                    download_name = _build_video_download_name(
                        task.get("video_subject"),
                        video_index + 1,
                        len(video_files),
                    )
                    with open(url, "rb") as video_file:
                        st.download_button(
                            download_label,
                            data=video_file,
                            file_name=download_name,
                            mime=mimetypes.guess_type(url)[0] or "video/mp4",
                            key=f"download_generated_video_{task_id}_{video_index}",
                            icon=":material/download:",
                            on_click="ignore",
                            use_container_width=True,
                        )
    except Exception as exc:
        logger.exception(
            f"failed to render generated video preview: task_id={task_id}, "
            f"video_files={video_files}, error={exc}"
        )

    _render_generation_logs(task_id)
    if (
        state == const.TASK_STATE_COMPLETE
        and st.session_state.get("handled_generation_task_id") != task_id
    ):
        # Fragments may repeatedly render the same completed task. Regardless of auto-open settings,
        # handle the completion event only once per task to prevent popping file managers or writing logs multiple times.
        st.session_state["handled_generation_task_id"] = task_id
        if config.ui.get("open_task_folder_on_completion", True):
            open_task_folder(task_id)
        logger.info(f"{tr('Video Generation Completed')}: task_id={task_id}")


@st.fragment(run_every=webui_task.TASK_LOG_REFRESH_INTERVAL_SECONDS)
def _render_running_generation_task(task_id):
    """Poll only while the task is running; switch to static results upon completion to stop unnecessary periodic refreshes."""
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.exception(
            f"failed to query WebUI generation task: task_id={task_id}, error={exc}"
        )
        st.error(tr("Video Generation Failed"))
        return

    state = _normalize_task_state((task or {}).get("state"))
    if state in {const.TASK_STATE_COMPLETE, const.TASK_STATE_FAILED}:
        _remove_active_generation_task(task_id)
        # The full page script now contains no long-running generation logic, so it can safely rerun and switch
        # results to static rendering. This avoids keeping a 2-second polling fragment active in the browser permanently after task completion.
        st.rerun(scope="app")

    _render_generation_task_snapshot(task_id, task)


def _render_current_generation_task():
    """Restore queryable UI below the generate button for the most recent task submitted from the current page."""
    task_id = st.session_state.get("current_generation_task_id", "")
    if not task_id:
        return

    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.exception(
            f"failed to query current WebUI task: task_id={task_id}, error={exc}"
        )
        st.error(tr("Video Generation Failed"))
        return

    state = _normalize_task_state((task or {}).get("state"))
    if state in {const.TASK_STATE_COMPLETE, const.TASK_STATE_FAILED}:
        _remove_active_generation_task(task_id)
        _render_generation_task_snapshot(task_id, task)
        return

    _render_running_generation_task(task_id)


def get_llm_provider_tips(provider_id, **kwargs):
    # LLM provider explanatory copy follows the `llm_provider_tips.<provider_id>` rule.
    # Newly added providers only need copy added in locale files; omitting tips hides the callout box,
    # preventing Main.py from accumulating hardcoded bilingual explanations.
    provider = get_llm_provider(provider_id)
    if provider is None:
        return ""

    # Provider configuration notes are currently maintained in standard Chinese and English templates; other UI
    # languages fall back to English to avoid out-of-sync copies across locales. Once a language completes full translation,
    # it can be added to independent maintenance here.
    ui_language = st.session_state.get("ui_language", "en")
    tips_language = ui_language if ui_language in {"zh", "en"} else "en"
    tips = (
        locales.get(tips_language, {}).get("Translation", {}).get(provider.tips_key, "")
    )
    if not tips:
        return tips

    service_endpoint = provider.preferred_service_endpoint(
        prefer_international=tips_language == "en"
    )
    api_key_url = (
        service_endpoint.api_key_url
        if service_endpoint
        else provider.effective_api_key_url()
    )
    format_context = {
        "api_key_url": api_key_url,
        "default_model": provider.default_model,
        "default_base_url": (
            service_endpoint.base_url
            if service_endpoint
            else provider.effective_default_base_url
        ),
        "model_docs_url": (
            service_endpoint.model_docs_url
            if service_endpoint and service_endpoint.model_docs_url
            else provider.effective_model_docs_url(
                prefer_international=tips_language == "en"
            )
        ),
        **{
            f"default_{field.config_suffix}": field.default_value
            for field in provider.extra_fields
        },
        **kwargs,
    }
    try:
        return tips.format(**format_context)
    except Exception as e:
        logger.warning(f"format llm provider tips failed: {provider_id}, {e}")
        return tips


def format_llm_connection_error(provider_id, base_url, error):
    """Supplement determinable authentication errors with configuration check advice while preserving raw response."""
    error_text = str(error or "").strip()
    normalized_error = error_text.lower()
    authentication_markers = (
        "401",
        "authentication",
        "invalid api key",
        "invalid_api_key",
        "unauthorized",
    )
    provider = get_llm_provider(provider_id)
    if provider is None or not provider.service_endpoints or not any(
        marker in normalized_error for marker in authentication_markers
    ):
        return error_text

    message = tr_optional(
        provider.authentication_error_key,
        fallback_language="en",
    )
    if not message:
        return error_text
    return message.format(base_url=base_url or "-", error=error_text)


def get_llm_provider_label(provider):
    return tr_optional(provider.label_key) or provider.default_label


def get_tts_provider_tips(provider_id):
    # TTS configuration notes follow the same maintenance strategy as LLM providers: maintain Chinese and English only;
    # other UI languages fall back to English to prevent desynchronization.
    ui_language = st.session_state.get("ui_language", "en")
    tips_language = ui_language if ui_language in {"zh", "en"} else "en"
    return (
        locales.get(tips_language, {})
        .get("Translation", {})
        .get(f"tts_provider_tips.{provider_id}", "")
    )


def localized_widget_key(name, *parts):
    # Some Streamlit selectboxes use stable keys to remember selections, but display text comes from locale.
    # Including the language code in the key forces widget recreation on language switch, preventing selected items from showing old text.
    language = st.session_state.get("ui_language", config.ui.get("language", ""))
    suffix_parts = [name, language, *[str(part) for part in parts if part]]
    return "_".join(suffix_parts)


def stable_selectbox(label, options, default_value, key, format_func=None, **kwargs):
    # Streamlit 1.59 is sensitive to selectbox state reuse: without a fixed key or when options are temporary indices,
    # reruns easily overwrite selections with recomputed indices, causing first clicks to be ignored.
    # This helper uses stable business values as true options and stores them in session_state;
    # display copy is transformed solely via format_func, isolating selection state from translations or option ordering.
    options = list(options)
    if not options:
        raise ValueError(f"selectbox options cannot be empty: {key}")

    if default_value not in options:
        default_value = options[0]

    widget_key = localized_widget_key(key)
    selected_value = st.session_state.get(widget_key)
    accepts_custom_value = bool(kwargs.get("accept_new_options"))
    has_valid_custom_value = (
        accepts_custom_value
        and isinstance(selected_value, str)
        and bool(selected_value.strip())
    )
    if selected_value not in options and not has_valid_custom_value:
        # If upstream options change (e.g. switching TTS provider changes the voice list), the old value becomes invalid.
        # Initialize session_state before widget creation, then let the key manage state without passing index.
        # This prevents Streamlit from overwriting newly selected values with recomputed indices during rerun.
        st.session_state[widget_key] = default_value

    if format_func is None:
        format_func = str

    return st.selectbox(
        label,
        options=options,
        format_func=format_func,
        key=widget_key,
        **kwargs,
    )


# Streamlit native selectbox lacks HTML optgroup support. Wrap native <select>/<optgroup> using Components v2
# included in 1.59 without external frontend dependencies, preserving native keyboard navigation, accessibility semantics,
# and mobile selection UX. The component passes only fixed business values and translated text, avoiding arbitrary HTML injection into innerHTML.
_GROUPED_SELECT_COMPONENT = st.components.v2.component(
    "mpt_grouped_select",
    html="""
        <div class="mpt-grouped-select">
            <div class="mpt-grouped-select__label-row">
                <label class="mpt-grouped-select__label"></label>
                <button class="mpt-grouped-select__settings" type="button"></button>
            </div>
            <div class="mpt-grouped-select__control">
                <select></select>
            </div>
        </div>
    """,
    css="""
        .mpt-grouped-select {
            width: 100%;
            color: var(--st-text-color);
            font-family: var(--st-font);
        }

        .mpt-grouped-select__label-row {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 0.45rem;
            margin-bottom: 0.35rem;
        }

        .mpt-grouped-select__label {
            font-size: 0.875rem;
            line-height: 1.25rem;
        }

        .mpt-grouped-select__settings {
            padding: 0;
            border: 0;
            background: transparent;
            color: var(--st-link-color);
            font: inherit;
            font-size: 0.8rem;
            line-height: 1.25rem;
            cursor: pointer;
        }

        .mpt-grouped-select__settings:hover {
            text-decoration: underline;
            text-underline-offset: 0.15rem;
        }

        .mpt-grouped-select__settings:focus-visible {
            border-radius: 0.2rem;
            outline: 2px solid var(--st-primary-color);
            outline-offset: 2px;
        }

        .mpt-grouped-select__control {
            position: relative;
        }

        .mpt-grouped-select__control::after {
            position: absolute;
            top: 50%;
            right: 1rem;
            width: 0.55rem;
            height: 0.55rem;
            border-right: 2px solid currentColor;
            border-bottom: 2px solid currentColor;
            content: "";
            pointer-events: none;
            transform: translateY(-70%) rotate(45deg);
        }

        .mpt-grouped-select select {
            width: 100%;
            min-height: 2.5rem;
            padding: 0.45rem 2.75rem 0.45rem 0.75rem;
            border: 1px solid color-mix(in srgb, currentColor 20%, transparent);
            border-radius: 0.5rem;
            outline: none;
            appearance: none;
            background: var(--st-secondary-background-color);
            color: inherit;
            font: inherit;
            cursor: pointer;
        }

        .mpt-grouped-select select:hover {
            border-color: color-mix(in srgb, currentColor 36%, transparent);
        }

        .mpt-grouped-select select:focus-visible {
            border-color: var(--st-primary-color);
            box-shadow: 0 0 0 1px var(--st-primary-color);
        }
    """,
    js="""
        export default function(component) {
            const { data, parentElement, setTriggerValue } = component;
            const label = parentElement.querySelector("label");
            const settings = parentElement.querySelector(".mpt-grouped-select__settings");
            const select = parentElement.querySelector("select");

            label.textContent = data.label;
            settings.textContent = data.settingsLabel;
            settings.hidden = !data.settingsLabel;
            select.id = data.controlId;
            label.htmlFor = data.controlId;
            select.setAttribute("aria-label", data.label);
            select.replaceChildren();

            for (const groupData of data.groups) {
                const group = document.createElement("optgroup");
                group.label = groupData.label;
                for (const optionData of groupData.options) {
                    const option = document.createElement("option");
                    option.value = optionData.value;
                    option.textContent = optionData.label;
                    group.appendChild(option);
                }
                select.appendChild(group);
            }

            select.value = data.value;
            const handleChange = () => {
                setTriggerValue("selected", select.value);
            };
            const handleSettings = () => {
                setTriggerValue("settings", true);
            };
            select.addEventListener("change", handleChange);
            settings.addEventListener("click", handleSettings);

            return () => {
                select.removeEventListener("change", handleChange);
                settings.removeEventListener("click", handleSettings);
            };
        }
    """,
)


def grouped_selectbox(
    label,
    groups,
    default_value,
    key,
    format_func=None,
    settings_label="",
    on_settings=None,
):
    """Render a single select dropdown with unselectable group headers and return a stable business value."""
    if format_func is None:
        format_func = str

    normalized_groups = []
    valid_values = []
    for group_label, options in groups:
        normalized_options = []
        for option in options:
            valid_values.append(option)
            normalized_options.append(
                {"value": option, "label": str(format_func(option))}
            )
        if normalized_options:
            normalized_groups.append(
                {"label": str(group_label), "options": normalized_options}
            )

    if not valid_values:
        raise ValueError(f"grouped selectbox options cannot be empty: {key}")
    if len(set(valid_values)) != len(valid_values):
        raise ValueError(f"grouped selectbox options must be unique: {key}")
    if default_value not in valid_values:
        default_value = valid_values[0]

    # Business selection is stored in the same session key as the legacy selectbox, avoiding forks in preset restoration
    # and language switching logic; the component itself uses an independent key to prevent collisions.
    widget_key = localized_widget_key(key)
    if widget_key not in st.session_state:
        st.session_state[widget_key] = default_value
    selected_value = st.session_state[widget_key]
    if selected_value not in valid_values:
        selected_value = default_value
        st.session_state[widget_key] = selected_value

    result = _GROUPED_SELECT_COMPONENT(
        key=f"{widget_key}_component",
        data={
            "label": label,
            "settingsLabel": settings_label,
            # Explicitly associate visible labels with native selects. The component key combines fixed business names
            # and language codes, remaining unique within the page to facilitate click-to-focus without recreating frontend state on rerun.
            "controlId": f"{widget_key}_control",
            "value": selected_value,
            "groups": normalized_groups,
        },
        on_selected_change=lambda: None,
        on_settings_change=on_settings or (lambda: None),
    )
    changed_value = getattr(result, "selected", None)
    if changed_value in valid_values and changed_value != selected_value:
        st.session_state[widget_key] = changed_value
        # When Components v2 emits an event in the current script run, frontend data still reflects pre-event state.
        # Rerun immediately so the component and controls dependent on video_source receive new values simultaneously,
        # preventing old data from briefly overriding the dropdown and requiring a second user selection.
        st.rerun()

    return selected_value


def sync_script_order_concat_mode():
    """Lock concat mode to sequential when script order matching is enabled, restoring the original choice when disabled."""
    widget_key = localized_widget_key("video_concat_mode_select")
    previous_key = "video_concat_mode_before_script_order_match"
    match_script_order = bool(st.session_state.get("match_materials_to_script", False))

    if match_script_order:
        current_mode = st.session_state.get(widget_key, VideoConcatMode.random.value)
        if current_mode != VideoConcatMode.sequential.value:
            st.session_state[previous_key] = current_mode
        st.session_state[widget_key] = VideoConcatMode.sequential.value
        return

    previous_mode = st.session_state.pop(previous_key, None)
    if previous_mode in {
        VideoConcatMode.sequential.value,
        VideoConcatMode.random.value,
    }:
        st.session_state[widget_key] = previous_mode


def reset_script_system_prompt():
    """Reset system prompt in advanced script settings to current version default."""
    st.session_state["custom_system_prompt"] = llm.DEFAULT_SCRIPT_SYSTEM_PROMPT


def reset_subtitle_settings():
    """Restore default values for WebUI subtitle widgets and persisted configurations."""
    defaults = DEFAULT_SUBTITLE_SETTINGS
    st.session_state["subtitle_enabled_checkbox"] = defaults["subtitle_enabled"]
    _set_stable_widget_value("font_name_select", defaults["font_name"])
    _set_stable_widget_value("subtitle_position_select", defaults["subtitle_position"])
    _set_stable_widget_value(
        "subtitle_display_mode_select", defaults["subtitle_display_mode"]
    )
    _set_stable_widget_value(
        "subtitle_animation_select", defaults["subtitle_animation"]
    )
    st.session_state["custom_position_input"] = str(defaults["custom_position"])
    st.session_state["font_color_picker"] = defaults["text_fore_color"]
    st.session_state["font_size_slider"] = defaults["font_size"]
    st.session_state["stroke_color_picker"] = defaults["stroke_color"]
    st.session_state["stroke_width_slider"] = defaults["stroke_width"]
    st.session_state["subtitle_background_enabled_checkbox"] = defaults[
        "subtitle_background_enabled"
    ]
    st.session_state["subtitle_background_color_picker"] = defaults[
        "subtitle_background_color"
    ]
    st.session_state["rounded_subtitle_background_checkbox"] = defaults[
        "rounded_subtitle_background"
    ]

    # Synchronize persisted UI options to ensure defaults remain intact upon page refresh after restoration.
    for key in (
        "subtitle_enabled",
        "font_name",
        "subtitle_position",
        "subtitle_display_mode",
        "subtitle_animation",
        "custom_position",
        "text_fore_color",
        "font_size",
        "stroke_color",
        "stroke_width",
        "subtitle_background_enabled",
        "subtitle_background_color",
        "rounded_subtitle_background",
    ):
        if key in defaults:
            _set_runtime_config("ui", key, defaults[key])


@st.dialog(tr("Final Prompt Preview"), width="large")
def render_script_prompt_preview(prompt):
    """Display the full script generation prompt to be sent to the LLM."""
    st.code(prompt, language="markdown", wrap_lines=True)


def stable_segmented_control(
    label, options, default_value, key, format_func=None, **kwargs
):
    """Create a single-select segmented control using stable business values to prevent display copy overriding state on language switch."""
    options = list(options)
    if not options:
        raise ValueError(f"segmented control options cannot be empty: {key}")

    if default_value not in options:
        default_value = options[0]

    widget_key = localized_widget_key(key)
    if st.session_state.get(widget_key) not in options:
        st.session_state[widget_key] = default_value

    return st.segmented_control(
        label,
        options=options,
        selection_mode="single",
        required=True,
        format_func=format_func or str,
        key=widget_key,
        **kwargs,
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_groq_model_ids(api_key: str, base_url: str) -> list[str]:
    if not api_key:
        return []

    normalized_base_url = (
        (base_url or "https://api.groq.com/openai/v1").strip().rstrip("/")
    )
    models_url = f"{normalized_base_url}/models"

    try:
        response = trace_http_call(
            "GET", requests.get,
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])

        model_ids = []
        for item in data:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    model_ids.append(model_id.strip())

        return sorted(set(model_ids))
    except Exception as e:
        logger.warning(f"failed to fetch groq models: {e}")
        return []


def _get_material_api_keys(config_key):
    """Normalize material API keys in configuration into WebUI-editable strings."""
    api_keys = config.app.get(config_key, [])
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    return ", ".join(api_keys)


def _save_material_api_keys(config_key, value):
    """Save comma-separated material API keys, allowing users to explicitly clear legacy configurations."""
    normalized_value = value.replace(" ", "")
    _set_runtime_config(
        "app",
        config_key,
        normalized_value.split(",") if normalized_value else [],
    )


def _format_file_size(size_bytes):
    """Format bytes into compact size text suitable for display in settings."""
    size = float(max(0, size_bytes))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


@st.cache_data(ttl=30, show_spinner=False)
def _get_video_cache_stats_data(max_age_days=None):
    """
    Short-cache directory statistics to prevent frequent file scans during regular widget interactions in settings dialog.

    Cache keys include cleanup age in days, scanning each range only once; explicitly cleared after manual refresh
    or cleanup completion, and the 30-second TTL does not affect secondary scans during actual deletion.

    Only plain dicts are cached here rather than VideoCacheStats instances: st.cache_data pickles return values,
    and pickle serializes custom classes by 'module_name + class_name' reference, verifying that the unpickled class
    matches the instance class object. When local source files change (including virus scanners or indexers on Windows),
    Streamlit source watcher reloads modules in sys.modules; dialogs (inheriting fragment behavior without rebuilding
    top-level Main.py references) retain old class objects, causing pickle to raise
    PicklingError: it's not the same object as ..., wrapped by Streamlit into
    UnserializableReturnValueError (Streamlit issue #14593).
    Plain dicts serialize by value without class references, making them immune to module reloads.
    """
    stats = cache_manager.get_video_cache_stats(max_age_days=max_age_days)
    # Field names match VideoCacheStats exactly for direct unpacking in _get_video_cache_stats below;
    # update this field mapping if VideoCacheStats adds or removes fields in the future.
    return {
        "file_count": stats.file_count,
        "total_size": stats.total_size,
        "oldest_mtime": stats.oldest_mtime,
        "newest_mtime": stats.newest_mtime,
    }


def _get_video_cache_stats(max_age_days=None) -> cache_manager.VideoCacheStats:
    """Reconstruct plain dict data into VideoCacheStats outside cache boundary, preserving attribute access syntax."""

    # Reconstruct using the currently active cache_manager module; even if reloaded, it merely builds
    # a lightweight dataclass without triggering pickle class identity checks.
    return cache_manager.VideoCacheStats(
        **_get_video_cache_stats_data(max_age_days=max_age_days)
    )


def _render_cache_management_settings(panel):
    """Render cache statistics, preview, and safe cleanup actions for default online video materials."""
    with panel:
        cleanup_message = st.session_state.pop("video_cache_cleanup_message", None)
        if cleanup_message:
            message_type, message = cleanup_message
            if message_type == "success":
                st.success(message)
            else:
                st.warning(message)

        st.caption(tr("Video Cache Directory"))
        st.code(cache_manager.video_cache_dir(), language="text")

        total_stats = _get_video_cache_stats()
        metric_count, metric_size, metric_oldest = st.columns(3)
        metric_count.metric(tr("Cache File Count"), total_stats.file_count)
        metric_size.metric(
            tr("Cache Total Size"), _format_file_size(total_stats.total_size)
        )
        oldest_text = (
            datetime.fromtimestamp(total_stats.oldest_mtime).strftime("%Y-%m-%d")
            if total_stats.oldest_mtime is not None
            else "-"
        )
        metric_oldest.metric(tr("Oldest Cache Date"), oldest_text)

        st.caption(tr("Video Cache Management Help"))
        cleanup_options = (30, 7, 90, None)
        cleanup_labels = {
            30: tr("Cache Older Than 30 Days"),
            7: tr("Cache Older Than 7 Days"),
            90: tr("Cache Older Than 90 Days"),
            None: tr("All Video Cache"),
        }
        max_age_days = st.selectbox(
            tr("Cache Cleanup Range"),
            options=cleanup_options,
            format_func=lambda value: cleanup_labels[value],
            key="video_cache_cleanup_range",
        )
        cleanup_preview = _get_video_cache_stats(max_age_days=max_age_days)
        st.info(
            tr("Cache Cleanup Preview").format(
                count=cleanup_preview.file_count,
                size=_format_file_size(cleanup_preview.total_size),
            )
        )

        confirm_nonce = st.session_state.get("video_cache_cleanup_confirm_nonce", 0)
        confirmed = st.checkbox(
            tr("Confirm Cache Cleanup"),
            key=f"video_cache_cleanup_confirm_{confirm_nonce}",
        )
        refresh_col, open_col, cleanup_col = st.columns(3)
        if refresh_col.button(
            tr("Refresh Cache Stats"),
            key="refresh_video_cache_stats",
            use_container_width=True,
            icon=":material/refresh:",
        ):
            _get_video_cache_stats_data.clear()
            st.rerun(scope="fragment")

        if open_col.button(
            tr("Open Cache Directory"),
            key="open_video_cache_directory",
            use_container_width=True,
            icon=":material/folder_open:",
        ):
            webbrowser.open(Path(cache_manager.video_cache_dir()).as_uri())

        cleanup_disabled = not confirmed or cleanup_preview.file_count == 0
        if cleanup_col.button(
            tr("Clean Cache Now"),
            key="clean_video_cache_now",
            type="primary",
            disabled=cleanup_disabled,
            use_container_width=True,
            icon=":material/delete_sweep:",
        ):
            result = cache_manager.clean_video_cache(max_age_days=max_age_days)
            message_key = (
                "Cache Cleanup Completed With Failures"
                if result.failed_count
                else "Cache Cleanup Completed"
            )
            st.session_state["video_cache_cleanup_message"] = (
                "warning" if result.failed_count else "success",
                tr(message_key).format(
                    count=result.deleted_count,
                    size=_format_file_size(result.deleted_size),
                    failed=result.failed_count,
                ),
            )
            # Streamlit disallows modifying same-named session_state after widget instantiation.
            # Incrementing nonce ensures the next fragment rerun creates an unchecked widget,
            # preventing dangerous confirmation states from lingering after cleanup.
            st.session_state["video_cache_cleanup_confirm_nonce"] = confirm_nonce + 1
            _get_video_cache_stats_data.clear()
            st.rerun(scope="fragment")


# -----------------------------------------------------------------------------
# Settings preset export/import and API key backup
# -----------------------------------------------------------------------------


def _is_credential_config_key(key):
    """Determine whether a configuration item name represents credentials."""
    return str(key).endswith(CREDENTIAL_KEY_SUFFIXES)


def _is_backup_config_key(section_name, key):
    """Credentials and their companion configuration items both belong to key backups."""
    if _is_credential_config_key(key):
        return True
    if key in CREDENTIAL_COMPANION_KEYS.get(section_name, ()):
        return True
    return key in NON_LLM_COMPANION_KEYS.get(section_name, ())


def _credential_widget_state_keys(section_name, key):
    """
    Return all Streamlit widget keys corresponding to a credential configuration item.

    Password inputs use keys; in Streamlit, session_state takes precedence over widget value parameters.
    Residual widget states must be cleared after backup restoration; otherwise, pages continue showing old keys
    and write them back on the next rerun, making restoration appear ineffective. Return the default key
    and all aliases when multiple panels share the same key.
    """
    if section_name == "app":
        default_widget_key = f"{key}_input"
    else:
        default_widget_key = f"{section_name}_{key}_input"
    return (
        default_widget_key,
        *CREDENTIAL_WIDGET_STATE_ALIASES.get((section_name, key), ()),
    )


def _normalize_backup_value(value):
    """Normalize backup values, discarding empty strings and empty lists to prevent overwriting with blanks during restoration."""
    if isinstance(value, list):
        items = [
            str(item).strip()
            for item in value
            if isinstance(item, (str, int, float)) and str(item).strip()
        ]
        return items or None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).strip()
        return text or None
    return None


def _collect_key_backup(config_sections):
    """Collect all configured keys and their companion settings from runtime configuration sections."""
    backup = {}
    for section_name, section in config_sections.items():
        if section_name in KEY_BACKUP_EXCLUDED_SECTIONS:
            continue
        entries = {}
        for key, value in section.items():
            if not _is_backup_config_key(section_name, key):
                continue
            normalized_value = _normalize_backup_value(value)
            if normalized_value is not None:
                entries[key] = normalized_value
        if entries:
            backup[section_name] = entries
    return backup


def _count_backup_keys(backup):
    """Count configuration items in backup for UI feedback and disabling empty exports."""
    return sum(len(entries) for entries in backup.values())


def _build_key_backup_payload(config_sections, app_version):
    """Construct key backup file payload."""
    return {
        "schema": KEY_BACKUP_SCHEMA,
        "version": KEY_BACKUP_VERSION,
        "app_version": str(app_version),
        "keys": _collect_key_backup(config_sections),
    }


def _load_transfer_payload(raw_bytes, schema, version):
    """
    Parse export file and verify it originates from the same version of this feature.

    Users may upload arbitrary JSON. Accept only files declaring the correct schema and version,
    halting error feedback at the import entry point rather than writing unrecognized content into config or widgets.
    Decoded with utf-8-sig to support JSON saved with BOM by Windows editors.
    """
    if len(raw_bytes) > MAX_SETTINGS_TRANSFER_BYTES:
        raise ValueError("settings import exceeds the 2 MB limit")
    payload = json.loads(raw_bytes.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("exported file must contain a JSON object")
    if payload.get("schema") != schema:
        raise ValueError(f"unexpected schema: {payload.get('schema')!r}")
    if payload.get("version") != version:
        raise ValueError(f"unsupported version: {payload.get('version')!r}")
    return payload


def _parse_key_backup(raw_bytes, config_sections):
    """
    Parse key backup file, retaining only sections and config items recognized by the current version.

    Backup files can be manually edited or exported from newer versions. Ignore unknown sections
    or non-credential items to prevent imports from altering settings unrelated to credentials.
    """
    payload = _load_transfer_payload(raw_bytes, KEY_BACKUP_SCHEMA, KEY_BACKUP_VERSION)
    keys = payload.get("keys")
    if not isinstance(keys, dict):
        raise ValueError("key backup file has no keys object")

    restored = {}
    for section_name, entries in keys.items():
        if section_name not in config_sections:
            continue
        if section_name in KEY_BACKUP_EXCLUDED_SECTIONS:
            continue
        if not isinstance(entries, dict):
            continue
        section_entries = {}
        for key, value in entries.items():
            if not _is_backup_config_key(section_name, key):
                continue
            normalized_value = _normalize_backup_value(value)
            if normalized_value is not None:
                section_entries[key] = normalized_value
        if section_entries:
            restored[section_name] = section_entries

    if not restored:
        raise ValueError("key backup file contains no restorable keys")
    return restored


def _build_settings_preset_payload(params, app_version):
    """Construct generation parameter preset payload."""
    preset_params = {
        key: value
        for key, value in params.items()
        if key not in PRESET_EXCLUDED_PARAM_KEYS
    }
    if params.get("bgm_type") == "preset" and params.get("bgm_file"):
        try:
            builtin_bgm_path = bgm_service.resolve_builtin_bgm_file(
                str(params["bgm_file"])
            )
        except ValueError:
            # Custom files are local machine resources and cannot enter portable settings presets.
            # Preserve existing exclusions in edge cases to avoid exporting absolute paths or non-existent UUIDs.
            pass
        else:
            preset_params["bgm_file"] = Path(builtin_bgm_path).name
    return {
        "schema": SETTINGS_PRESET_SCHEMA,
        "version": SETTINGS_PRESET_VERSION,
        "app_version": str(app_version),
        "params": preset_params,
    }


def _parse_settings_preset(raw_bytes):
    """
    Parse preset file and delegate validation to VideoParams.

    Presets may be generated on another machine or edited manually. Routing through model validation
    reuses existing value range constraints, rejecting invalid presets at import rather than failing during task generation.
    """
    payload = _load_transfer_payload(
        raw_bytes, SETTINGS_PRESET_SCHEMA, SETTINGS_PRESET_VERSION
    )
    preset_params = payload.get("params")
    if not isinstance(preset_params, dict):
        raise ValueError("settings preset file has no params object")

    params_input = {
        key: value
        for key, value in preset_params.items()
        if key not in PRESET_EXCLUDED_PARAM_KEYS
    }
    if preset_params.get("bgm_type") == "preset" and preset_params.get("bgm_file"):
        # Settings presets can only restore built-in songs present in the current version. The service layer
        # rejects both directory separators and user uploads to prevent imported files reading arbitrary local paths via preview.
        builtin_bgm_path = bgm_service.resolve_builtin_bgm_file(
            str(preset_params["bgm_file"])
        )
        params_input["bgm_file"] = Path(builtin_bgm_path).name
    # video_subject is required by VideoParams, but presets are permitted to save style settings only.
    params_input.setdefault("video_subject", "")
    return VideoParams.model_validate(params_input).model_dump(mode="json")


def _apply_key_backup(restored_keys):
    """Write parsed keys back to runtime configuration and clear residual states of corresponding widgets."""
    restored_count = 0
    for section_name, entries in restored_keys.items():
        for key, value in entries.items():
            _set_runtime_config(section_name, key, value)
            for widget_key in _credential_widget_state_keys(section_name, key):
                st.session_state.pop(widget_key, None)
            restored_count += 1
    # ElevenLabs voice lists are cached per key; refetch when switching to a different backup.
    for cache_key in list(st.session_state.keys()):
        if str(cache_key).startswith("elevenlabs_voices_"):
            del st.session_state[cache_key]
    return restored_count


def _apply_pending_settings_preset():
    """Apply imported presets before rendering any widgets."""
    preset_params = st.session_state.pop("settings_preset_payload", None)
    if not preset_params:
        return False

    _apply_restored_params(preset_params)
    logger.info("applied imported settings preset")
    return True


def _render_settings_transfer(params):
    """Render export and import entry points for generation parameter presets."""
    with st.expander(tr("Settings Preset"), expanded=False):
        st.caption(tr("Settings Preset Help"))
        preset_payload = _build_settings_preset_payload(
            params.model_dump(mode="json"), config.project_version
        )
        st.download_button(
            tr("Export Settings"),
            data=json.dumps(preset_payload, ensure_ascii=False, indent=2).encode(
                "utf-8"
            ),
            file_name=SETTINGS_PRESET_FILE_NAME,
            mime="application/json",
            use_container_width=True,
            key="export_settings_preset_button",
            icon=":material/download:",
        )
        uploaded_preset = st.file_uploader(
            tr("Import Settings"),
            type=["json"],
            key="settings_preset_uploader",
        )
        if uploaded_preset is None:
            return
        # Uploaded files reappear on every subsequent rerun. Record processed file identifiers
        # to prevent users' manual widget adjustments from being repeatedly overwritten by the same preset.
        if st.session_state.get("settings_preset_file_id") == uploaded_preset.file_id:
            return

        st.session_state["settings_preset_file_id"] = uploaded_preset.file_id
        try:
            preset_params = _parse_settings_preset(uploaded_preset.getvalue())
        except Exception as e:
            logger.warning(f"failed to import settings preset: {e}")
            st.error(tr("Settings Preset Import Failed"))
            return

        st.session_state["settings_preset_payload"] = preset_params
        st.rerun()


def _render_key_backup_settings(panel):
    """Render export and restore entry points for API key backups."""
    with panel:
        backup_message = st.session_state.pop("key_backup_message", None)
        if backup_message:
            message_type, message = backup_message
            if message_type == "success":
                st.success(message)
            else:
                st.error(message)

        st.caption(tr("Key Backup Help"))
        st.warning(tr("Key Backup Warning"))

        backup_payload = _build_key_backup_payload(
            _RUNTIME_CONFIG_SECTIONS, config.project_version
        )
        backup_key_count = _count_backup_keys(backup_payload["keys"])
        st.caption(tr("Key Backup Summary").format(count=backup_key_count))
        st.download_button(
            tr("Export Keys"),
            data=json.dumps(backup_payload, ensure_ascii=False, indent=2).encode(
                "utf-8"
            ),
            file_name=KEY_BACKUP_FILE_NAME,
            mime="application/json",
            disabled=backup_key_count == 0,
            use_container_width=True,
            key="export_key_backup_button",
            icon=":material/download:",
        )

        uploaded_backup = st.file_uploader(
            tr("Import Keys"),
            type=["json"],
            key="key_backup_uploader",
        )
        if uploaded_backup is None:
            return
        if st.session_state.get("key_backup_file_id") == uploaded_backup.file_id:
            return

        st.session_state["key_backup_file_id"] = uploaded_backup.file_id
        try:
            restored_keys = _parse_key_backup(
                uploaded_backup.getvalue(), _RUNTIME_CONFIG_SECTIONS
            )
        except Exception as e:
            logger.warning(f"failed to import key backup: {e}")
            st.session_state["key_backup_message"] = (
                "error",
                tr("Key Restore Failed"),
            )
        else:
            restored_count = _apply_key_backup(restored_keys)
            _save_runtime_config()
            logger.info(f"restored keys from backup file: count={restored_count}")
            st.session_state["key_backup_message"] = (
                "success",
                tr("Keys Restored").format(count=restored_count),
            )
        # The TTS key input on the main page also needs to read restored configurations, so trigger a full-page refresh.
        # The settings dialog open state is stored in session_state, reopening automatically after the refresh.
        st.rerun(scope="app")


# -----------------------------------------------------------------------------
# Settings and prompt dialogs
# -----------------------------------------------------------------------------


def _render_logging_settings():
    level, log_requests = get_logging_options()
    with st.expander(tr("Diagnostic Logging")):
        st.caption(tr("Diagnostic Logging Help"))
        with st.form("diagnostic_logging_form"):
            selected_level = st.selectbox(
                tr("Log Level"), LOG_LEVELS, index=LOG_LEVELS.index(level),
                key="diagnostic_log_level",
            )
            selected_trace = st.checkbox(
                tr("Trace API Requests"), value=log_requests,
                key="diagnostic_log_requests",
                help=tr("Trace API Requests Help"),
            )
            apply = st.form_submit_button(tr("Apply Logging Settings"))
        if apply:
            # Diagnostics do not alter generation parameters and take effect immediately; persistence goes through non-blocking queue.
            config.update_logging_options(selected_level, selected_trace)
            _save_runtime_config()
            st.success(tr("Logging Settings Applied"))


# Settings are infrequent operations; use medium-sized Dialogs to avoid consuming vertical space on the main page,
# while constraining reading line length to prevent sparse layouts on wide displays.
# Dialogs inherit fragment behavior so inner widget interactions redraw only the dialog; config is saved at function end,
# triggering full-page sync on close via callback to ensure generation pipelines read the latest provider and UI settings.
@st.dialog(
    tr("Settings"),
    width="medium",
    on_dismiss=_dismiss_settings_dialog,
)
def _render_settings_dialog():
    with st.container():
        # Legacy hide_config was only used to hide the old basic settings panel. With a dedicated settings entry point,
        # this value has no user-facing meaning; migrate to false to prevent stale configs affecting future versions.
        _set_runtime_config("app", "hide_config", False)
        settings_tab_labels = [
            tr("LLM Settings Tab"),
            tr("Material API Tab"),
            tr("Auto-Publish Settings"),
            tr("Interface Settings Tab"),
            tr("Key Backup Tab"),
            tr("Cache Management Tab"),
        ]
        settings_tab_targets = {
            "llm": tr("LLM Settings Tab"),
            "material": tr("Material API Tab"),
        }
        settings_tabs_key = localized_widget_key("settings_dialog_tabs")
        target_tab = st.session_state.pop("settings_dialog_target_tab", None)
        if target_tab in settings_tab_targets:
            # st.tabs uses display labels as state values. The entry button stores only stable business IDs;
            # writing current language labels here achieves exact navigation compatible with language switching.
            st.session_state[settings_tabs_key] = settings_tab_targets[target_tab]

        (
            middle_config_panel,
            right_config_panel,
            publish_config_panel,
            left_config_panel,
            key_backup_panel,
            cache_config_panel,
        ) = st.tabs(
            settings_tab_labels,
            key=settings_tabs_key,
            on_change="rerun",
        )

        with publish_config_panel:
            st.write(tr("Automatically publish generated videos to social media using upload-post.com"))
            st.info(
                tr("Upload-Post Setup Guide").format(
                    api_keys_url=UPLOAD_POST_API_KEYS_URL,
                    manage_users_url=UPLOAD_POST_MANAGE_USERS_URL,
                )
            )

            is_enabled = config.app.get("upload_post_enabled", False)
            is_auto = config.app.get("upload_post_auto_upload", False)

            # The two keys are independent: enabled allows external pipelines to call Upload-Post,
            # while auto_upload decides whether to publish automatically upon rendering. Merging into one checkbox
            # would rewrite enabled to False just by opening the dialog when the two keys differ.
            upload_post_enabled = st.checkbox(
                tr("Enable Upload-Post Integration"),
                value=is_enabled,
                key="upload_post_enabled_checkbox"
            )
            if upload_post_enabled != is_enabled:
                _set_runtime_config("app", "upload_post_enabled", upload_post_enabled)

            upload_post_auto_upload = st.checkbox(
                tr("Enable Auto-Publish"),
                value=is_auto,
                key="upload_post_auto_upload_checkbox"
            )
            if upload_post_auto_upload != is_auto:
                _set_runtime_config("app", "upload_post_auto_upload", upload_post_auto_upload)

            upload_post_api_key = st.text_input(
                tr("Upload-Post API Key"),
                value=config.app.get("upload_post_api_key", ""),
                type="password",
                help=tr("Upload-Post API Key Help").format(
                    api_keys_url=UPLOAD_POST_API_KEYS_URL
                ),
                key="upload_post_api_key_input"
            )
            if upload_post_api_key != config.app.get("upload_post_api_key", ""):
                _set_runtime_config("app", "upload_post_api_key", upload_post_api_key)

            upload_post_username = st.text_input(
                tr("Upload-Post Profile Username"),
                value=config.app.get("upload_post_username", ""),
                help=tr("Upload-Post Profile Username Help").format(
                    manage_users_url=UPLOAD_POST_MANAGE_USERS_URL
                ),
                key="upload_post_username_input"
            )
            if upload_post_username != config.app.get("upload_post_username", ""):
                _set_runtime_config("app", "upload_post_username", upload_post_username)

            upload_post_platforms = st.multiselect(
                tr("Platforms"),
                options=["tiktok", "instagram", "youtube"],
                default=config.app.get("upload_post_platforms", ["tiktok", "instagram"]),
                help="Select platforms to publish to",
                key="upload_post_platforms_multiselect"
            )
            if upload_post_platforms != config.app.get("upload_post_platforms", ["tiktok", "instagram"]):
                _set_runtime_config("app", "upload_post_platforms", upload_post_platforms)

            if "youtube" in upload_post_platforms:
                yt_status_options = ["public", "private", "unlisted"]
                yt_saved = config.app.get("upload_post_youtube_privacy_status", "public")
                if yt_saved not in yt_status_options:
                    yt_saved = "public"
                upload_post_youtube_privacy_status = st.selectbox(
                    tr("YouTube Privacy Status"),
                    options=yt_status_options,
                    index=yt_status_options.index(yt_saved),
                    key="upload_post_youtube_privacy_status_selectbox"
                )
                if upload_post_youtube_privacy_status != config.app.get("upload_post_youtube_privacy_status", "public"):
                    _set_runtime_config("app", "upload_post_youtube_privacy_status", upload_post_youtube_privacy_status)

                # Audience declarations only affect YouTube publishing without altering generated content or requests for other platforms.
                # Use real boolean options to avoid passing display copy or strings as API parameters.
                saved_audience = config.app.get("upload_post_youtube_made_for_kids", False)
                audience_labels = {False: tr("Not Made for Kids"), True: tr("Made for Kids")}
                made_for_kids = st.selectbox(
                    tr("YouTube Audience"),
                    options=[False, True],
                    # Keep invalid configurations unselected rather than silently switching to non-child declarations on open.
                    index=int(saved_audience) if isinstance(saved_audience, bool) else None,
                    format_func=audience_labels.get,
                    help=tr("YouTube Audience Help"),
                    key="upload_post_youtube_made_for_kids_selectbox",
                )
                if isinstance(made_for_kids, bool):
                    _set_runtime_config("app", "upload_post_youtube_made_for_kids", made_for_kids)

        # Left panel - Logging settings
        with left_config_panel:
            hide_log = st.checkbox(
                tr("Hide Log"),
                value=config.ui.get("hide_log", False),
                key="hide_log_checkbox",
            )
            _set_runtime_config("ui", "hide_log", hide_log)
            _render_logging_settings()

        _render_cache_management_settings(cache_config_panel)
        # Key restoration writes back to config and clears password widget states; must execute before rendering widgets below.
        _render_key_backup_settings(key_backup_panel)

        # Middle panel - LLM settings

        with middle_config_panel:
            # Dropdown order, default labels, and stable provider IDs all come from Registry;
            # locale only overrides display copy, removing the duplicate provider list from Main.py.
            llm_provider_ids = [
                provider.provider_id for provider in LLM_PROVIDER_REGISTRY
            ]
            llm_provider_labels = {
                provider.provider_id: get_llm_provider_label(provider)
                for provider in LLM_PROVIDER_REGISTRY
            }
            saved_llm_provider = config.app.get(
                "llm_provider", DEFAULT_LLM_PROVIDER_ID
            ).lower()
            if saved_llm_provider not in llm_provider_ids:
                saved_llm_provider = DEFAULT_LLM_PROVIDER_ID

            llm_provider = stable_selectbox(
                tr("LLM Provider"),
                options=llm_provider_ids,
                default_value=saved_llm_provider,
                key="llm_provider_select",
                format_func=lambda provider_id: llm_provider_labels[provider_id],
            )
            # Display configuration form and provider instructions side by side to reduce text wrapping in narrow columns
            # while maximizing horizontal space in the basic settings panel.
            llm_form_panel, llm_help_panel = st.columns(
                [0.9, 1.1],
                gap="large",
                vertical_alignment="top",
            )
            llm_helper = llm_help_panel.container()
            _set_runtime_config("app", "llm_provider", llm_provider)
            llm_provider_spec = get_llm_provider(llm_provider)
            if llm_provider_spec is None:
                # Under normal circumstances all options come from Registry; retain explicit error for diagnosing
                # corrupted session state or omissions in new provider integrations.
                raise RuntimeError(f"unsupported llm provider: {llm_provider}")

            llm_api_key = config.app.get(llm_provider_spec.config_key("api_key"), "")
            configured_llm_base_url = config.app.get(
                llm_provider_spec.config_key("base_url"), ""
            )
            llm_default_base_url = llm_provider_spec.effective_default_base_url
            llm_base_url = configured_llm_base_url or llm_default_base_url
            llm_model_name = llm_provider_spec.resolve_model_name(
                config.app.get(llm_provider_spec.config_key("model_name"), "")
            )

            provider_tip_context = {}
            selected_service_endpoint = None
            if llm_provider_spec.service_endpoints:
                # Providers like Kimi use separate account systems for China and International regions.
                # Let users choose the service region and allow Registry to synchronize API portals and Base URLs,
                # preventing manual pairing errors. Existing empty Base URL configs keep China defaults;
                # only new configs without keys recommend portal URLs matching the current UI language.
                selected_service_endpoint = (
                    llm_provider_spec.select_service_endpoint(
                        configured_llm_base_url,
                        has_api_key=bool(str(llm_api_key).strip()),
                        prefer_international=(
                            st.session_state.get("ui_language", "en") != "zh"
                        ),
                    )
                )
                endpoint_options = [
                    endpoint.endpoint_id
                    for endpoint in llm_provider_spec.service_endpoints
                ] + [CUSTOM_LLM_ENDPOINT_ID]
                default_endpoint_id = (
                    selected_service_endpoint.endpoint_id
                    if selected_service_endpoint
                    else CUSTOM_LLM_ENDPOINT_ID
                )
                endpoint_labels = {
                    endpoint.endpoint_id: (
                        tr_optional(
                            llm_provider_spec.endpoint_label_key(endpoint.endpoint_id),
                            fallback_language="en",
                        )
                        or endpoint.default_label
                    )
                    for endpoint in llm_provider_spec.service_endpoints
                }
                endpoint_labels[CUSTOM_LLM_ENDPOINT_ID] = (
                    tr_optional("Custom API Endpoint", fallback_language="en")
                    or "Custom API Endpoint"
                )
                with llm_form_panel:
                    selected_endpoint_id = stable_selectbox(
                        tr_optional(
                            llm_provider_spec.endpoint_selector_label_key,
                            fallback_language="en",
                        )
                        or tr("API Platform"),
                        options=endpoint_options,
                        default_value=default_endpoint_id,
                        key=f"{llm_provider}_service_endpoint_select",
                        format_func=lambda endpoint_id: endpoint_labels[endpoint_id],
                        help=(
                            tr_optional(
                                llm_provider_spec.endpoint_selector_help_key,
                                fallback_language="en",
                            )
                            or None
                        ),
                    )
                selected_service_endpoint = next(
                    (
                        endpoint
                        for endpoint in llm_provider_spec.service_endpoints
                        if endpoint.endpoint_id == selected_endpoint_id
                    ),
                    None,
                )
                if selected_service_endpoint:
                    llm_base_url = selected_service_endpoint.base_url
                    provider_tip_context.update(
                        {
                            "api_key_url": selected_service_endpoint.api_key_url,
                            "default_base_url": selected_service_endpoint.base_url,
                            "model_docs_url": selected_service_endpoint.model_docs_url,
                        }
                    )
                else:
                    # Custom mode only preserves explicitly saved endpoints without disguising standard regions as custom values.
                    # Empty inputs are not persisted, reverting to compatible defaults on the next run.
                    llm_base_url = str(configured_llm_base_url or "").strip()

            if llm_provider == "ollama":
                llm_default_base_url = config.get_default_ollama_base_url()
                if not llm_base_url:
                    llm_base_url = llm_default_base_url
                docker_hint = ""
                if config.is_running_in_container():
                    docker_hint = tr_optional(
                        "llm_provider_tips.ollama.docker_hint",
                        fallback_language="en",
                    )
                provider_tip_context["docker_hint"] = docker_hint

            tips = get_llm_provider_tips(llm_provider, **provider_tip_context)
            if tips:
                with llm_helper:
                    st.info(tips)

            st_llm_api_key = llm_api_key
            if llm_provider_spec.show_api_key:
                st_llm_api_key = llm_form_panel.text_input(
                    tr("API Key"),
                    value=llm_api_key,
                    type="password",
                    key=f"{llm_provider}_api_key_input",
                )

            st_llm_base_url = llm_base_url
            if llm_provider_spec.show_base_url:
                st_llm_base_url = llm_form_panel.text_input(
                    tr("Base Url"),
                    value=llm_base_url,
                    key=(
                        f"{llm_provider}_base_url_"
                        f"{selected_service_endpoint.endpoint_id}_input"
                        if selected_service_endpoint
                        else f"{llm_provider}_base_url_custom_input"
                    ),
                    disabled=selected_service_endpoint is not None,
                )
            st_llm_model_name = ""
            if llm_provider == "groq":
                effective_api_key = st_llm_api_key or llm_api_key
                effective_base_url = st_llm_base_url or llm_base_url
                groq_models = get_groq_model_ids(
                    api_key=effective_api_key,
                    base_url=effective_base_url,
                )

                if groq_models:
                    selected_index = 0
                    if llm_model_name in groq_models:
                        selected_index = groq_models.index(llm_model_name)

                    st_llm_model_name = llm_form_panel.selectbox(
                        tr("Model Name"),
                        options=groq_models,
                        index=selected_index,
                        key="groq_model_name_select",
                    )
                else:
                    st_llm_model_name = llm_form_panel.text_input(
                        tr("Model Name"),
                        value=llm_model_name,
                        key="groq_model_name_input",
                    )
                    if effective_api_key:
                        llm_form_panel.caption(tr("Groq Model List Load Failed"))
                    else:
                        llm_form_panel.caption(
                            tr("Groq API Key Required for Model List")
                        )
            else:
                st_llm_model_name = llm_form_panel.text_input(
                    tr("Model Name"),
                    value=llm_model_name,
                    key=f"{llm_provider}_model_name_input",
                )
            # The input displays Registry defaults while configuration stores only user overrides.
            # This ensures non-customizing users automatically track upstream default model and Base URL updates.
            _set_runtime_config(
                "app",
                llm_provider_spec.config_key("api_key"),
                st_llm_api_key,
            )
            _set_runtime_config(
                "app",
                llm_provider_spec.config_key("base_url"),
                normalize_provider_override(
                    st_llm_base_url,
                    llm_default_base_url,
                ),
            )
            _set_runtime_config(
                "app",
                llm_provider_spec.config_key("model_name"),
                normalize_provider_override(
                    st_llm_model_name,
                    llm_provider_spec.default_model,
                ),
            )

            # Provider-specific fields are also declared by the Registry (e.g. Account ID for Cloudflare AI Gateway);
            # adding similar fields in the future requires no extra branching in Main.py.
            for field in llm_provider_spec.extra_fields:
                field_config_key = llm_provider_spec.config_key(field.config_suffix)
                field_value = llm_form_panel.text_input(
                    tr(field.label_key),
                    value=(config.app.get(field_config_key, "") or field.default_value),
                    type="password" if field.secret else "default",
                    key=f"{llm_provider}_{field.config_suffix}_input",
                )
                _set_runtime_config(
                    "app",
                    field_config_key,
                    normalize_provider_override(
                        field_value,
                        field.default_value,
                    ),
                )

            if llm_provider == "kimi_code" and middle_config_panel.open:
                from app.services import kimi_code_oauth
                from webui.kimi_login import render_kimi_login

                # Provide a dedicated container for nested fragments so polling redraws do not shift following widgets.
                with llm_form_panel.container():
                    render_kimi_login(
                        kimi_code_oauth.resolve_region(st_llm_base_url), tr
                    )

            if llm_form_panel.button(
                tr("Test LLM Connection"),
                key="test_llm_connection_button",
                use_container_width=True,
                type="secondary",
                icon=":material/network_check:",
            ):
                with config.try_runtime_config_lock() as lock_acquired:
                    if not lock_acquired:
                        llm_form_panel.warning(tr("Runtime Configuration Busy"))
                    else:
                        with llm_form_panel.spinner(tr("Testing LLM Connection")):
                            connection_ok, connection_error, connection_elapsed = (
                                llm.test_connection()
                            )

                if not lock_acquired:
                    connection_ok = None
                elif connection_ok:
                    llm_form_panel.success(
                        tr("LLM Connection Test Succeeded").format(
                            provider=llm_provider_labels[llm_provider],
                            model=st_llm_model_name or "-",
                            elapsed=f"{connection_elapsed:.2f}",
                        )
                    )
                else:
                    connection_error = format_llm_connection_error(
                        llm_provider,
                        st_llm_base_url,
                        connection_error,
                    )
                    llm_form_panel.error(
                        tr("LLM Connection Test Failed").format(error=connection_error)
                    )

        # Right panel - API key settings
        with right_config_panel:
            # Material providers are grouped by 'Search Stock Materials / AI Video Generation / AI Image Generation',
            # avoiding an unorganized long list as providers expand.
            # Grouping only adjusts display hierarchy without altering config keys, ensuring existing users read config.toml seamlessly.
            with st.container(border=True):
                st.markdown(f"#### {tr('Stock Video APIs')}")
                st.caption(tr("Stock Video APIs Help"))

                pexels_api_key = _get_material_api_keys("pexels_api_keys")
                pixabay_api_key = _get_material_api_keys("pixabay_api_keys")
                coverr_api_key = _get_material_api_keys("coverr_api_keys")
                pexels_api_key = st.text_input(
                    tr("Pexels API Key"),
                    value=pexels_api_key,
                    type="password",
                    key="pexels_api_keys_input",
                )
                _save_material_api_keys("pexels_api_keys", pexels_api_key)

                pixabay_api_key = st.text_input(
                    tr("Pixabay API Key"),
                    value=pixabay_api_key,
                    type="password",
                    key="pixabay_api_keys_input",
                )
                _save_material_api_keys("pixabay_api_keys", pixabay_api_key)

                coverr_api_key = st.text_input(
                    tr("Coverr API Key"),
                    value=coverr_api_key,
                    type="password",
                    key="coverr_api_keys_input",
                )
                _save_material_api_keys("coverr_api_keys", coverr_api_key)

            with st.container(border=True):
                st.markdown(f"#### {tr('AI Video Generation APIs')}")
                st.caption(tr("AI Video Generation APIs Help"))

                # Video generation providers prioritize sponsors, following the internal order in VIDEO_SOURCE_GROUPS:
                # MetaSo, OFox, Sonilo, Volcengine.
                st.markdown(f"**{tr('Metaso MiniMax H3')}**")
                metaso_api_key = st.text_input(
                    tr("Metaso MiniMax API Key"),
                    value=str(
                        config.app.get("metaso_minimax_api_key", "") or ""
                    ).strip(),
                    type="password",
                    help=tr("Metaso MiniMax API Key Help"),
                    key="metaso_minimax_api_key_input",
                )
                _set_runtime_config(
                    "app", "metaso_minimax_api_key", metaso_api_key.strip()
                )
                configured_metaso_base_url = str(
                    config.app.get(
                        "metaso_minimax_base_url",
                        metaso_minimax.DEFAULT_BASE_URL,
                    )
                    or metaso_minimax.DEFAULT_BASE_URL
                ).strip()
                metaso_base_url = st.text_input(
                    tr("Metaso MiniMax Base URL"),
                    value=(
                        ""
                        if configured_metaso_base_url == metaso_minimax.DEFAULT_BASE_URL
                        else configured_metaso_base_url
                    ),
                    placeholder=metaso_minimax.DEFAULT_BASE_URL,
                    key="metaso_minimax_base_url_input",
                )
                _set_runtime_config(
                    "app",
                    "metaso_minimax_base_url",
                    metaso_base_url.strip() or metaso_minimax.DEFAULT_BASE_URL,
                )
                configured_metaso_resolution = (
                    str(
                        config.app.get(
                            "metaso_minimax_resolution",
                            metaso_minimax.DEFAULT_RESOLUTION,
                        )
                    )
                    .strip()
                    .upper()
                )
                metaso_resolution_options = sorted(
                    metaso_minimax.SUPPORTED_RESOLUTIONS,
                    key=lambda value: value != metaso_minimax.DEFAULT_RESOLUTION,
                )
                resolution_is_valid = (
                    configured_metaso_resolution
                    in metaso_minimax.SUPPORTED_RESOLUTIONS
                )
                if not resolution_is_valid:
                    # Resolution directly affects billing. Preserve manual config errors and prompt users to select,
                    # rather than silently upgrading to higher-priced 2K upon opening settings.
                    st.error(
                        tr("Metaso MiniMax Invalid Resolution").format(
                            value=configured_metaso_resolution,
                            supported=", ".join(metaso_resolution_options),
                        )
                    )
                metaso_resolution = st.selectbox(
                    tr("Metaso MiniMax Resolution"),
                    options=metaso_resolution_options,
                    index=(
                        metaso_resolution_options.index(configured_metaso_resolution)
                        if resolution_is_valid
                        else None
                    ),
                    key="metaso_minimax_resolution_input",
                    help=tr("Metaso MiniMax Resolution Help"),
                    placeholder=tr("Select Metaso MiniMax Resolution"),
                )
                if metaso_resolution is not None:
                    _set_runtime_config(
                        "app", "metaso_minimax_resolution", metaso_resolution
                    )

                st.divider()
                st.markdown("**OfoxAI**")
                st.caption(f"[OfoxAI]({OFOX_REFERRAL_URL}) · {tr('OFox AI Video Help')}")
                ofox_api_key = st.text_input(
                    tr("OFox API Key"),
                    value=str(config.app.get("ofox_api_key", "") or ""),
                    type="password",
                    key="ofox_api_key_input",
                )
                _set_runtime_config("app", "ofox_api_key", ofox_api_key.strip())
                ofox_model = st.text_input(
                    tr("OFox Text-to-Video Model"),
                    value=str(
                        config.app.get(
                            "ofox_text_to_video_model",
                            ofox.DEFAULT_MODEL_ID,
                        )
                        or ofox.DEFAULT_MODEL_ID
                    ),
                    key="ofox_text_to_video_model_input",
                )
                _set_runtime_config(
                    "app", "ofox_text_to_video_model", ofox_model.strip()
                )
                configured_ofox_base_url = str(
                    config.app.get("ofox_base_url", ofox.DEFAULT_BASE_URL)
                    or ofox.DEFAULT_BASE_URL
                ).strip()
                ofox_base_url = st.text_input(
                    tr("OFox Base URL"),
                    value=(
                        ""
                        if configured_ofox_base_url == ofox.DEFAULT_BASE_URL
                        else configured_ofox_base_url
                    ),
                    placeholder=ofox.DEFAULT_BASE_URL,
                    key="ofox_base_url_input",
                )
                _set_runtime_config(
                    "app",
                    "ofox_base_url",
                    ofox_base_url.strip() or ofox.DEFAULT_BASE_URL,
                )
                ofox_vendor_options = [
                    (tr("OFox Vendor BytePlus"), "byteplus"),
                    (tr("OFox Vendor Volcengine"), "volcengine"),
                    (tr("OFox Vendor Auto"), ""),
                ]
                configured_ofox_vendor = str(
                    config.app.get("ofox_provider", ofox.DEFAULT_PROVIDER_TYPE)
                    or ""
                ).strip()
                if configured_ofox_vendor not in {
                    value for _, value in ofox_vendor_options
                }:
                    # Retain user selections when another vendor is manually pinned in config.toml,
                    # preventing the settings dropdown from overwriting it back to default on open.
                    ofox_vendor_options.append(
                        (configured_ofox_vendor, configured_ofox_vendor)
                    )
                selected_ofox_vendor = stable_selectbox(
                    tr("OFox Upstream Vendor"),
                    options=[value for _, value in ofox_vendor_options],
                    default_value=configured_ofox_vendor,
                    key="ofox_provider_select",
                    format_func=lambda value: dict(
                        (v, label) for label, v in ofox_vendor_options
                    )[value],
                    help=tr("OFox Upstream Vendor Help"),
                )
                _set_runtime_config("app", "ofox_provider", selected_ofox_vendor)

                st.divider()
                st.markdown(f"**{tr('Shengsuan Cloud AI Video')}**")
                app_config_snapshot = config.snapshot_config_with_pending(config.app)
                if (
                    str(app_config_snapshot.get("llm_provider", "") or "").lower()
                    == "shengsuanyun"
                ):
                    # When LLM provider is Sonilo, video generation directly reuses the same key,
                    # removing an ambiguous separate input field.
                    st.caption(tr("Shengsuan Cloud API Key Reused"))
                else:
                    configured_loomloom_token = str(
                        app_config_snapshot.get("loomloom_api_token", "") or ""
                    ).strip()
                    loomloom_api_token = st.text_input(
                        tr("Shengsuan Cloud API Key"),
                        value=configured_loomloom_token,
                        type="password",
                        key="loomloom_api_token_input",
                        help=tr("Shengsuan Cloud API Key Help"),
                        placeholder=tr("Shengsuan Cloud API Key Placeholder"),
                    ).strip()
                    _set_runtime_config(
                        "app", "loomloom_api_token", loomloom_api_token
                    )

                st.divider()
                seedance_api_key_value = str(
                    config.app.get("volcengine_seedance_api_key", "") or ""
                ).strip()
                shared_ark_api_key = str(
                    config.app.get("volcengine_api_key", "") or ""
                ).strip()
                environment_ark_api_key = os.getenv(
                    "VOLCENGINE_ARK_API_KEY", ""
                ).strip()
                seedance_reuses_llm_key = bool(
                    not seedance_api_key_value
                    and not environment_ark_api_key
                    and shared_ark_api_key
                )
                seedance_title = f"**{tr('Volcano Engine Seedance')}**"
                if seedance_reuses_llm_key:
                    # Only LLM key reuse cannot be directly inferred from the input; keep this tip to prevent users
                    # thinking duplicate entry is required, omitting it for ordinary configs.
                    seedance_title += f" :blue[{tr('Reusing LLM API Key')}]"
                st.markdown(seedance_title)
                seedance_api_key = st.text_input(
                    tr("Volcano Engine Ark API Key"),
                    value=seedance_api_key_value,
                    type="password",
                    help=tr("Volcano Engine Ark API Key Help"),
                    key="volcengine_seedance_api_key_input",
                )
                _set_runtime_config(
                    "app", "volcengine_seedance_api_key", seedance_api_key.strip()
                )
                configured_seedance_model = str(
                    config.app.get(
                        "volcengine_seedance_model",
                        volcengine_seedance.DEFAULT_MODEL_ID,
                    )
                    or volcengine_seedance.DEFAULT_MODEL_ID
                ).strip()
                seedance_model = st.text_input(
                    tr("Volcano Engine Seedance Model"),
                    # Built-in defaults are shown via placeholders, while user-customized model or endpoint IDs
                    # are rendered and persisted as true values.
                    value=(
                        ""
                        if configured_seedance_model
                        == volcengine_seedance.DEFAULT_MODEL_ID
                        else configured_seedance_model
                    ),
                    placeholder=volcengine_seedance.DEFAULT_MODEL_ID,
                    key="volcengine_seedance_model_input",
                )
                _set_runtime_config(
                    "app",
                    "volcengine_seedance_model",
                    seedance_model.strip() or volcengine_seedance.DEFAULT_MODEL_ID,
                )
                configured_seedance_base_url = str(
                    config.app.get(
                        "volcengine_seedance_base_url",
                        volcengine_seedance.DEFAULT_BASE_URL,
                    )
                    or volcengine_seedance.DEFAULT_BASE_URL
                ).strip()
                seedance_base_url = st.text_input(
                    tr("Volcano Engine Ark Base URL"),
                    value=(
                        ""
                        if configured_seedance_base_url
                        == volcengine_seedance.DEFAULT_BASE_URL
                        else configured_seedance_base_url
                    ),
                    placeholder=volcengine_seedance.DEFAULT_BASE_URL,
                    key="volcengine_seedance_base_url_input",
                )
                _set_runtime_config(
                    "app",
                    "volcengine_seedance_base_url",
                    seedance_base_url.strip() or volcengine_seedance.DEFAULT_BASE_URL,
                )

                st.divider()
                wavespeed_api_key = _get_material_api_keys("wavespeed_api_keys")
                st.markdown("**WaveSpeed**")
                wavespeed_api_key = st.text_input(
                    tr("WaveSpeed API Key"),
                    value=wavespeed_api_key,
                    type="password",
                    key="wavespeed_api_keys_input",
                )
                _save_material_api_keys("wavespeed_api_keys", wavespeed_api_key)

                st.divider()
                st.markdown(f"**{tr('MuAPI AI Video')}**")
                st.caption(tr("MuAPI AI Video Help"))
                muapi_api_key = st.text_input(
                    tr("MuAPI API Key"),
                    value=str(config.app.get("muapi_api_key", "") or ""),
                    type="password",
                    help=tr("MuAPI API Key Help"),
                    key="muapi_api_key_input",
                )
                _set_runtime_config("app", "muapi_api_key", muapi_api_key.strip())
                configured_muapi_base_url = str(
                    config.app.get("muapi_base_url", muapi.DEFAULT_BASE_URL)
                    or muapi.DEFAULT_BASE_URL
                ).strip()
                muapi_base_url = st.text_input(
                    tr("MuAPI Base URL"),
                    value=(
                        ""
                        if configured_muapi_base_url == muapi.DEFAULT_BASE_URL
                        else configured_muapi_base_url
                    ),
                    placeholder=muapi.DEFAULT_BASE_URL,
                    key="muapi_base_url_input",
                    help=tr("MuAPI Base URL Help"),
                )
                _set_runtime_config(
                    "app",
                    "muapi_base_url",
                    muapi_base_url.strip() or muapi.DEFAULT_BASE_URL,
                )
                configured_muapi_endpoint = str(
                    config.app.get("muapi_video_endpoint", muapi.DEFAULT_ENDPOINT)
                    or muapi.DEFAULT_ENDPOINT
                ).strip()
                muapi_endpoint = st.text_input(
                    tr("MuAPI Video Endpoint"),
                    value=(
                        ""
                        if configured_muapi_endpoint == muapi.DEFAULT_ENDPOINT
                        else configured_muapi_endpoint
                    ),
                    placeholder=muapi.DEFAULT_ENDPOINT,
                    key="muapi_video_endpoint_input",
                    help=tr("MuAPI Video Endpoint Help"),
                )
                _set_runtime_config(
                    "app",
                    "muapi_video_endpoint",
                    muapi_endpoint.strip() or muapi.DEFAULT_ENDPOINT,
                )
                configured_muapi_resolution = str(
                    config.app.get("muapi_resolution", muapi.DEFAULT_RESOLUTION)
                    or muapi.DEFAULT_RESOLUTION
                ).strip()
                muapi_resolution = st.text_input(
                    tr("MuAPI Resolution"),
                    value=(
                        ""
                        if configured_muapi_resolution == muapi.DEFAULT_RESOLUTION
                        else configured_muapi_resolution
                    ),
                    placeholder=muapi.DEFAULT_RESOLUTION,
                    key="muapi_resolution_input",
                    help=tr("MuAPI Resolution Help"),
                )
                _set_runtime_config(
                    "app",
                    "muapi_resolution",
                    muapi_resolution.strip() or muapi.DEFAULT_RESOLUTION,
                )


            with st.container(border=True):
                st.markdown(f"#### {tr('AI Image Generation APIs')}")
                st.caption(tr("AI Image Generation APIs Help"))
                st.markdown(f"**{tr('OpenAI Compatible Text-to-Image')}**")

                openai_image_base_url = st.text_input(
                    tr("OpenAI Image Base URL"),
                    value=str(config.app.get("openai_image_base_url", "") or ""),
                    placeholder="https://api.openai.com/v1",
                    key="openai_image_base_url_input",
                )
                _set_runtime_config(
                    "app", "openai_image_base_url", openai_image_base_url.strip()
                )

                openai_image_api_key = _get_material_api_keys(
                    "openai_image_api_keys"
                )
                openai_image_api_key = st.text_input(
                    tr("OpenAI Image API Key"),
                    value=openai_image_api_key,
                    type="password",
                    help=tr("OpenAI Image API Key Help"),
                    key="openai_image_api_keys_input",
                )
                _save_material_api_keys(
                    "openai_image_api_keys", openai_image_api_key
                )

                openai_image_model = st.text_input(
                    tr("OpenAI Image Model"),
                    value=str(config.app.get("openai_image_model", "") or ""),
                    placeholder="gpt-image-2",
                    key="openai_image_model_input",
                )
                _set_runtime_config(
                    "app", "openai_image_model", openai_image_model.strip()
                )
                # Only display reference values without setting official OpenAI endpoints as default configuration.
                # Compatible services lack uniform Base URLs and model IDs; leaving blanks prevents unintentional
                # calls to paid official APIs and preserves legacy configs.
                st.caption(tr("OpenAI Image Configuration Example"))

                with st.expander(
                    tr("OpenAI Image Advanced Settings"), expanded=False
                ):
                    openai_image_size = st.text_input(
                        tr("OpenAI Image Size"),
                        value=str(config.app.get("openai_image_size", "") or ""),
                        placeholder="1024x1536",
                        help=tr("OpenAI Image Size Help"),
                        key="openai_image_size_input",
                    )
                    _set_runtime_config(
                        "app", "openai_image_size", openai_image_size.strip()
                    )

                    openai_image_prompt_template = st.text_input(
                        tr("OpenAI Image Prompt Template"),
                        value=str(
                            config.app.get("openai_image_prompt_template", "") or ""
                        ),
                        placeholder="cinematic photo of {term}, photorealistic",
                        help=tr("OpenAI Image Prompt Template Help"),
                        key="openai_image_prompt_template_input",
                    )
                    _set_runtime_config(
                        "app",
                        "openai_image_prompt_template",
                        openai_image_prompt_template.strip(),
                    )

    _save_runtime_config()


# -----------------------------------------------------------------------------
# Main generation forms: script, video, audio, and subtitle panels
# -----------------------------------------------------------------------------


def _create_loomloom_script_backend():
    """Create batch script client from current WebUI / config.toml configuration."""
    app_config_snapshot = config.snapshot_config_with_pending(config.app)
    settings = loomloom.LoomLoomSettings.from_mapping(app_config_snapshot)
    return loomloom.LoomLoomScriptBackend(settings)


def _create_loomloom_video_backend():
    """Create video client using project default SkillBot and currently active credentials."""
    app_config_snapshot = config.snapshot_config_with_pending(config.app)
    settings = loomloom.video_settings_from_mapping(app_config_snapshot)
    return loomloom.LoomLoomVideoBackend(settings)


def _effective_loomloom_api_token():
    """Read pending WebUI or config.toml Sonilo API key."""
    app_config_snapshot = config.snapshot_config_with_pending(config.app)
    return loomloom.resolve_api_token(app_config_snapshot)


def _effective_script_generation_backend():
    """Read script generation method including pending WebUI modifications."""
    app_config_snapshot = config.snapshot_config_with_pending(config.app)
    backend = str(
        app_config_snapshot.get("script_generation_backend", "local") or "local"
    ).strip()
    return backend if backend in {"local", "loomloom"} else "local"




def _script_generation_method_help(selected_backend):
    """Keep tooltip help text strictly synchronized with current script generation method."""
    if selected_backend != "loomloom":
        return tr("Script Generation Method Help")

    app_config_snapshot = config.snapshot_config_with_pending(config.app)
    guidance = [tr("LoomLoom Batch Script Generation Help")]
    if (
        str(app_config_snapshot.get("llm_provider", "") or "").strip().lower()
        == "shengsuanyun"
    ):
        guidance.append(tr("Shengsuan Cloud API Key Reused"))
    guidance.append(tr("Shengsuan Cloud API Key Link"))
    return "\n\n".join(guidance)


def _loomloom_video_scene_prompts(video_terms, subject, scene_count):
    """Generate limited scene descriptions based on material keywords for segment-by-segment video generation."""
    if isinstance(video_terms, str):
        terms = [
            term.strip() for term in re.split(r"[,，\n]", video_terms) if term.strip()
        ]
    elif isinstance(video_terms, list):
        terms = [
            str(term or "").strip() for term in video_terms if str(term or "").strip()
        ]
    else:
        terms = []
    fallback = str(subject or "").strip()
    if not terms and fallback:
        terms = [fallback]
    if not terms:
        return ()
    return tuple(
        (
            terms[index % len(terms)]
            if index < len(terms)
            else f"{terms[index % len(terms)]}; alternative camera angle {index + 1}"
        )
        for index in range(int(scene_count))
    )


def _loomloom_video_signature(batch, credential_fingerprint):
    """Include all billing parameters and credential signatures to force requoting upon parameter changes."""
    payload = {
        "inputRows": [dict(row) for row in batch.input_rows],
        "credentialFingerprint": str(credential_fingerprint or "").strip(),
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _loomloom_video_account_signature(token):
    """Service URLs and credentials isolate model catalogs and quotes; prevent cross-endpoint reuse of confirmed states."""
    values = config.snapshot_config_with_pending(config.app)
    base_url = str(values.get("loomloom_base_url") or loomloom.DEFAULT_BASE_URL).strip().rstrip("/")
    payload = json.dumps([base_url, str(token or "").strip()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_loomloom_video_capability(token, *, force=False):
    """Cache Profile by current credentials; retain most recent successful result on refresh failure."""
    normalized_token = str(token or "").strip()
    if not normalized_token:
        return None

    fingerprint = _loomloom_video_account_signature(normalized_token)
    if st.session_state.get("loomloom_video_capability_fingerprint") != fingerprint:
        st.session_state["loomloom_video_capability"] = None
        st.session_state["loomloom_video_capability_fingerprint"] = fingerprint
        st.session_state["loomloom_video_capability_load_attempt"] = ""
        st.session_state["loomloom_video_capability_error"] = ""

    should_load = force or (
        st.session_state.get("loomloom_video_capability_load_attempt") != fingerprint
    )
    if should_load:
        st.session_state["loomloom_video_capability_load_attempt"] = fingerprint
        try:
            capability = _create_loomloom_video_backend().resolve_video_capability()
        except (loomloom.LoomLoomError, ValueError) as exc:
            logger.warning(
                f"failed to load LoomLoom video capability: error={type(exc).__name__}"
            )
            st.session_state["loomloom_video_capability_error"] = str(exc)
        else:
            st.session_state["loomloom_video_capability"] = capability
            st.session_state["loomloom_video_capability_error"] = ""

    capability = st.session_state.get("loomloom_video_capability")
    return (
        capability if isinstance(capability, loomloom.LoomLoomVideoCapability) else None
    )


def _normalize_loomloom_model_identifier(value):
    """Normalize delimiters and casing of display names and model IDs for safe local pricing lookups."""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def _loomloom_video_model_price(model):
    """Return known model pricing tuple (dropdown short price, selected full reference price); returns empty strings for unknown models."""
    identifiers = {
        _normalize_loomloom_model_identifier(model.model_id),
        _normalize_loomloom_model_identifier(model.display_name),
    }
    for aliases, compact_price, detailed_price in LOOMLOOM_VIDEO_MODEL_PRICES:
        if identifiers.intersection(aliases):
            return compact_price, detailed_price
    return "", ""


def _format_loomloom_video_model_option(model):
    """Display short price to the right of model name to keep dropdown from becoming excessively wide."""
    compact_price, _ = _loomloom_video_model_price(model)
    return f"{model.display_name} · {compact_price}" if compact_price else model.display_name


def _effective_voice_rate_before_audio_panel():
    """Read current voice rate from widget state or config, as video panel renders before audio panel."""
    raw_rate = st.session_state.get(
        localized_widget_key("voice_rate_select"),
        config.ui.get("voice_rate", 1.0),
    )
    try:
        rate = float(raw_rate)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    return rate if math.isfinite(rate) and rate > 0 else 1.0


def _matching_full_voice_preview_duration(script, voice_rate):
    """Use actual duration from full voice preview only when copy, provider, voice, and rate remain unchanged."""
    cached = st.session_state.get("voice_preview_audio")
    if not isinstance(cached, dict) or cached.get("preview_type") != "full":
        return None
    script_digest = hashlib.sha256(str(script or "").encode("utf-8")).hexdigest()
    if cached.get("content_digest") != script_digest:
        return None

    current_tts_server = st.session_state.get(
        localized_widget_key("tts_server_select"),
        config.ui.get("tts_server", "azure-tts-v1"),
    )
    current_voice_name = st.session_state.get(
        localized_widget_key(f"speech_synthesis_select_{current_tts_server}"),
        config.ui.get("voice_name", ""),
    )
    try:
        cached_voice_rate = float(cached.get("voice_rate", 0))
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        cached.get("tts_server") != current_tts_server
        or cached.get("voice_name") != current_voice_name
        or not math.isfinite(cached_voice_rate)
        or not math.isclose(cached_voice_rate, voice_rate)
    ):
        return None

    duration = cached.get("duration")
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        return None
    return float(duration)


def _loomloom_video_coverage_plan(params):
    """Recommend material count based on actual or estimated narration duration; shortfalls are still filled by loop logic."""
    script = str(params.video_script or "").strip()
    if not script:
        return None

    voice_rate = _effective_voice_rate_before_audio_panel()
    actual_duration = _matching_full_voice_preview_duration(script, voice_rate)
    if actual_duration is not None:
        duration_min = duration_max = actual_duration
        basis_key = "AI Video Duration Basis Actual"
    else:
        estimated = _estimate_voiceover_duration_range(script, voice_rate)
        if not estimated:
            return None
        duration_min, duration_max = estimated
        basis_key = "AI Video Duration Basis Estimated"

    clip_duration = max(float(params.video_clip_duration or 1), 1.0)
    needed_min = max(math.ceil(duration_min / clip_duration), 1)
    needed_max = max(math.ceil(duration_max / clip_duration), needed_min)
    return {
        "script_digest": hashlib.sha256(script.encode("utf-8")).hexdigest(),
        "basis_key": basis_key,
        "duration_min": float(duration_min),
        "duration_max": float(duration_max),
        "clip_duration": clip_duration,
        "needed_min": needed_min,
        "needed_max": needed_max,
        # Recommendations conservatively cover the upper bound, but never exceed the server's paid task limit.
        "recommended_count": min(needed_max, loomloom.MAX_VIDEO_SCENES),
    }


def _format_numeric_range(minimum, maximum, digits=1):
    if math.isclose(float(minimum), float(maximum)):
        return f"{float(maximum):.{digits}f}"
    return f"{float(minimum):.{digits}f}–{float(maximum):.{digits}f}"


def _selected_loomloom_video_model(capability):
    """Return user selection if still present in current profile candidates without silent fallback."""
    selected_model_id = str(
        st.session_state.get("loomloom_video_model_id", "") or ""
    ).strip()
    eligible_model_ids = {model.model_id for model in capability.models}
    return selected_model_id if selected_model_id in eligible_model_ids else ""


def _current_loomloom_video_quote_context(params):
    """Construct video quote batch for default SkillBot based on current page parameters."""
    token = _effective_loomloom_api_token()
    fingerprint = _loomloom_video_account_signature(token) if token else ""
    capability = st.session_state.get("loomloom_video_capability")
    if (
        not isinstance(capability, loomloom.LoomLoomVideoCapability)
        or st.session_state.get("loomloom_video_capability_fingerprint") != fingerprint
    ):
        return None, ""
    model_id = _selected_loomloom_video_model(capability)
    scene_count = int(st.session_state.get("loomloom_video_scene_count", 1) or 1)
    prompts = _loomloom_video_scene_prompts(
        params.video_terms,
        params.video_subject or params.video_script,
        scene_count,
    )
    aspect_ratio = str(
        params.video_aspect.value
        if isinstance(params.video_aspect, VideoAspect)
        else params.video_aspect
    )
    if (
        not token
        or not model_id
        or not prompts
        or aspect_ratio not in capability.aspect_ratios
    ):
        return None, ""
    try:
        batch = _create_loomloom_video_backend().prepare_video_batch(
            subject=params.video_subject or params.video_script,
            scene_prompts=prompts,
            model_id=model_id,
            aspect_ratio=aspect_ratio,
        )
    except (loomloom.LoomLoomError, ValueError):
        return None, ""
    return batch, _loomloom_video_signature(batch, fingerprint)


def _retry_loomloom_video_quote():
    """Release failure lock on explicit user retry; previous payment confirmations are not reused."""
    st.session_state["loomloom_video_quote_error_signature"] = ""
    st.session_state["loomloom_video_quote_error"] = ""
    st.session_state["loomloom_video_confirm_charge"] = False


def _render_loomloom_video_settings(params):
    """Render quote, quote invalidation, and payment confirmation flow for default video SkillBot."""
    st.caption(tr("Shengsuan Cloud AI Video Help"))
    if (
        str(
            config.snapshot_config_with_pending(config.app).get("llm_provider", "")
            or ""
        ).lower()
        == "shengsuanyun"
    ):
        st.caption(tr("Shengsuan Cloud API Key Reused"))

    token = _effective_loomloom_api_token()

    refresh_models = st.button(
        tr("Refresh AI Video Models"),
        key="loomloom_refresh_video_models",
        use_container_width=True,
        disabled=not token,
    )
    capability = _load_loomloom_video_capability(token, force=refresh_models)
    capability_error = str(
        st.session_state.get("loomloom_video_capability_error", "") or ""
    ).strip()
    if capability_error:
        st.warning(tr("AI Video Model List Load Failed").format(error=capability_error))

    if capability is not None:
        models_by_id = {model.model_id: model for model in capability.models}
        selected_model_id = str(
            st.session_state.get("loomloom_video_model_id", "") or ""
        ).strip()
        if not selected_model_id:
            selected_model_id = capability.default_model_id
            st.session_state["loomloom_video_model_id"] = selected_model_id

        model_options = list(models_by_id)
        if selected_model_id not in models_by_id:
            # Preserve invalidated previous selections so users clearly see the state and reselect explicitly.
            # Silently switching to a new default model creates a mismatch with user expectations.
            model_options.insert(0, selected_model_id)

        selected_model_id = stable_selectbox(
            tr("AI Video Model"),
            options=model_options,
            default_value=selected_model_id,
            key="loomloom_video_model_select",
            format_func=lambda model_id: (
                _format_loomloom_video_model_option(models_by_id[model_id])
                if model_id in models_by_id
                else tr("Unavailable AI Video Model").format(model=model_id)
            ),
        )
        st.session_state["loomloom_video_model_id"] = selected_model_id
        if selected_model_id not in models_by_id:
            st.error(tr("Selected AI Video Model Unavailable"))
        else:
            _, detailed_price = _loomloom_video_model_price(
                models_by_id[selected_model_id]
            )
            if detailed_price:
                st.caption(
                    tr("AI Video Model Reference Price").format(price=detailed_price)
                )

        current_aspect_ratio = str(
            params.video_aspect.value
            if isinstance(params.video_aspect, VideoAspect)
            else params.video_aspect
        )
        if current_aspect_ratio not in capability.aspect_ratios:
            st.error(tr("Selected AI Video Ratio Unavailable"))

    coverage_plan = _loomloom_video_coverage_plan(params)
    pending_autofill_digest = str(
        st.session_state.get("loomloom_video_scene_autofill_digest", "") or ""
    )
    if (
        coverage_plan is not None
        and pending_autofill_digest == coverage_plan["script_digest"]
    ):
        # Recommend once only upon 'script freshly generated' or 'actual duration obtained from full preview'.
        # Do not override once consumed; user's subsequent manual adjustments are preserved.
        st.session_state["loomloom_video_scene_count"] = coverage_plan[
            "recommended_count"
        ]
        st.session_state["loomloom_video_scene_autofill_digest"] = ""

    scene_count = st.number_input(
        tr("AI Video Scene Count"),
        min_value=1,
        max_value=loomloom.MAX_VIDEO_SCENES,
        step=1,
        key="loomloom_video_scene_count",
    )
    _set_runtime_config("ui", "loomloom_video_scene_count", int(scene_count))
    if coverage_plan is not None:
        coverage_seconds = int(scene_count) * coverage_plan["clip_duration"]
        shortfall_min = max(
            coverage_plan["duration_min"] - coverage_seconds, 0.0
        )
        shortfall_max = max(
            coverage_plan["duration_max"] - coverage_seconds, 0.0
        )
        duration_basis = tr(coverage_plan["basis_key"]).format(
            duration=_format_numeric_range(
                coverage_plan["duration_min"], coverage_plan["duration_max"]
            )
        )
        coverage_message = tr("AI Video Material Coverage").format(
            basis=duration_basis,
            clip=_format_numeric_range(
                coverage_plan["clip_duration"], coverage_plan["clip_duration"]
            ),
            needed=_format_numeric_range(
                coverage_plan["needed_min"], coverage_plan["needed_max"], digits=0
            ),
            count=int(scene_count),
            coverage=_format_numeric_range(coverage_seconds, coverage_seconds),
            shortfall=_format_numeric_range(shortfall_min, shortfall_max),
        )
        if shortfall_max > 0:
            st.warning(coverage_message)
        else:
            st.caption(coverage_message)

    batch, input_signature = _current_loomloom_video_quote_context(params)
    if not token:
        st.warning(tr("Shengsuan Cloud API Key Required"))

    quote_result = st.session_state.get("loomloom_video_quote")
    quoted_batch = st.session_state.get("loomloom_video_batch")
    quote_is_current = bool(
        quote_result is not None
        and quoted_batch is not None
        and st.session_state.get("loomloom_video_input_signature") == input_signature
    )
    # Pause automatic requests after failure on the same parameter set to avoid blocking UI on service timeouts.
    # Signature includes account, endpoint, and billing inputs; re-quote only after parameter changes or manual retry.
    if st.session_state.get("loomloom_video_quote_error_signature") != input_signature:
        st.session_state["loomloom_video_quote_error_signature"] = ""
        st.session_state["loomloom_video_quote_error"] = ""
    quote_failed = bool(st.session_state.get("loomloom_video_quote_error"))
    # Quotes do not create billable tasks; actual execution requires explicit user confirmation.
    if token and batch is not None and not quote_is_current and not quote_failed:
        st.session_state["loomloom_video_confirm_charge"] = False
        try:
            quote_result = _create_loomloom_video_backend().quote(batch)
        except (loomloom.LoomLoomError, ValueError) as exc:
            logger.warning(f"failed to quote LoomLoom videos: error={exc}")
            st.session_state["loomloom_video_quote_error_signature"] = input_signature
            st.session_state["loomloom_video_quote_error"] = str(exc) or type(exc).__name__
        else:
            st.session_state["loomloom_video_batch"] = batch
            st.session_state["loomloom_video_quote"] = quote_result
            st.session_state["loomloom_video_input_signature"] = input_signature
            st.session_state["loomloom_video_client_request_id"] = (
                f"mpt-video-{uuid4()}"
            )
            st.session_state["loomloom_video_confirm_charge"] = False
            logger.info(
                "LoomLoom video quote ready: "
                f"tasks={quote_result.task_count}, currency={quote_result.currency}, "
                f"estimated_payable_t={quote_result.estimated_buyer_payable_t}"
            )

    if st.session_state.get("loomloom_video_quote_error"):
        st.error(st.session_state["loomloom_video_quote_error"])
        st.button(
            tr("Retry AI Video Quote"),
            key="loomloom_retry_video_quote",
            on_click=_retry_loomloom_video_quote,
        )

    quote_result = st.session_state.get("loomloom_video_quote")
    quoted_batch = st.session_state.get("loomloom_video_batch")
    if quote_result is not None and quoted_batch is not None:
        display_amount = (
            quote_result.estimated_buyer_payable_amount
            or f"{quote_result.estimated_buyer_payable_t} T"
        )
        if quote_result.estimated_buyer_payable_t == 0:
            st.warning(tr("AI Video Quote Estimate Incomplete"))
        else:
            st.success(
                tr(
                    "AI Video Quote Summary Singular"
                    if quote_result.task_count == 1
                    else "AI Video Quote Summary"
                ).format(
                    tasks=quote_result.task_count,
                    amount=display_amount,
                    currency=quote_result.currency,
                )
            )
        quote_is_current = (
            st.session_state.get("loomloom_video_input_signature") == input_signature
        )
        if not quote_is_current:
            st.warning(tr("LoomLoom Quote Changed Warning"))
        st.checkbox(
            tr("Confirm AI Video Charge"),
            key="loomloom_video_confirm_charge",
            help=tr("Confirm AI Video Charge Help"),
            disabled=not quote_is_current,
        )


def _loomloom_script_signature(
    *,
    subject,
    language,
    candidate_count,
    duration_seconds,
    style,
    credential_fingerprint,
):
    payload = {
        "subject": str(subject or "").strip(),
        "language": str(language or "auto").strip() or "auto",
        "candidateCount": int(candidate_count),
        "durationSeconds": int(duration_seconds),
        "style": str(style or "").strip(),
        "credentialFingerprint": str(credential_fingerprint or "").strip(),
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _render_local_script_generation(params):
    """Preserve MoneyPrinterTurbo's original local LLM script generation path."""
    if not st.button(
        tr("Generate Video Script and Keywords"),
        key="auto_generate_script",
        use_container_width=True,
        type="secondary",
        icon=":material/auto_awesome:",
    ):
        return

    if not params.video_subject:
        st.toast(tr("Please Enter the Video Subject First"))
        st.warning(tr("Please Enter the Video Subject First"))
        return

    with st.spinner(tr("Generating Video Script and Keywords")):

        def generate_script_and_terms(app_config_snapshot):
            script = llm.generate_script(
                video_subject=params.video_subject,
                language=params.video_language,
                paragraph_number=params.paragraph_number,
                video_script_prompt=params.video_script_prompt,
                custom_system_prompt=params.custom_system_prompt,
                app_config=app_config_snapshot,
            )
            terms = llm.generate_terms(
                params.video_subject,
                script,
                amount=8 if params.match_materials_to_script else 5,
                match_script_order=params.match_materials_to_script,
                app_config=app_config_snapshot,
            )
            return script, terms

        script, terms = _run_llm_read_operation(
            "generate_script_and_terms",
            generate_script_and_terms,
        )
        if "Error: " in script:
            st.error(tr(script))
        elif "Error: " in terms:
            st.error(tr(terms))
        else:
            st.session_state["video_script"] = script
            st.session_state["video_terms"] = ", ".join(terms)
            st.session_state["loomloom_video_scene_autofill_digest"] = (
                hashlib.sha256(script.strip().encode("utf-8")).hexdigest()
            )


def _render_loomloom_candidates():
    candidates = tuple(st.session_state.get("loomloom_script_candidates") or ())
    errors = tuple(st.session_state.get("loomloom_candidate_errors") or ())
    if errors:
        st.warning(
            tr("LoomLoom Candidate Errors").format(
                count=len(errors),
                details="; ".join(
                    f"#{error.row_index + 1}: {error.message}" for error in errors
                ),
            )
        )
    if not candidates:
        return

    selected_index = st.radio(
        tr("Choose Script Candidate"),
        options=list(range(len(candidates))),
        key="loomloom_selected_candidate",
        format_func=lambda index: (
            f"#{candidates[index].row_index + 1} {candidates[index].script[:80]}"
        ),
    )
    selected = candidates[selected_index]
    st.code(selected.script, language=None, wrap_lines=True)
    st.caption(", ".join(selected.video_terms))
    if st.button(
        tr("Use Selected Candidate"),
        key="loomloom_apply_candidate",
        type="primary",
        use_container_width=True,
    ):
        st.session_state["video_script"] = selected.script
        st.session_state["video_terms"] = ", ".join(selected.video_terms)
        # Consistent with standard LLM copy generation: recommend material count only once after applying candidates.
        st.session_state["loomloom_video_scene_autofill_digest"] = (
            hashlib.sha256(selected.script.strip().encode("utf-8")).hexdigest()
        )
        st.toast(tr("LoomLoom Candidate Applied"))


def _handle_loomloom_poll_error(run_id, exc):
    """Apply bounded backoff to script task polling errors, halting immediately on deterministic errors."""
    logger.warning(f"failed to poll LoomLoom run: run_id={run_id}, error={exc}")
    failure_count = int(st.session_state.get("loomloom_poll_failure_count", 0) or 0) + 1
    retryable = isinstance(exc, loomloom.LoomLoomAPIError) and exc.retryable
    if not retryable or failure_count >= LOOMLOOM_MAX_POLL_FAILURES:
        st.session_state["loomloom_run_error"] = str(exc)
        st.session_state["loomloom_poll_failure_count"] = 0
        st.session_state["loomloom_poll_retry_after"] = 0.0
        # Query failure does not mean remote billable task failure. Retain run_id and pause automatic polling
        # so users can continue checking the same task; discarding the ID and resubmitting risks duplicate charges.
        st.session_state["loomloom_poll_paused"] = True
        st.rerun(scope="app")
        return

    retry_delay = min(2**failure_count, 30)
    st.session_state["loomloom_poll_failure_count"] = failure_count
    st.session_state["loomloom_poll_retry_after"] = time.monotonic() + retry_delay
    st.warning(
        tr("LoomLoom Poll Retry Warning").format(
            attempt=failure_count,
            max_attempts=LOOMLOOM_MAX_POLL_FAILURES,
        )
    )


@st.fragment(run_every="2s")
def _render_loomloom_run_progress():
    run_id = str(st.session_state.get("loomloom_run_id", "") or "").strip()
    if not run_id or st.session_state.get("loomloom_poll_paused", False):
        return
    retry_after = float(st.session_state.get("loomloom_poll_retry_after", 0.0) or 0.0)
    retry_wait_seconds = max(0, int(math.ceil(retry_after - time.monotonic())))
    if retry_wait_seconds > 0:
        st.info(
            tr("LoomLoom Poll Retry Pending").format(
                seconds=retry_wait_seconds,
            )
        )
        return
    try:
        backend = _create_loomloom_script_backend()
        run = backend.get_run(run_id)
    except loomloom.LoomLoomError as exc:
        _handle_loomloom_poll_error(run_id, exc)
        return

    st.session_state["loomloom_run_status"] = run.status
    if run.status == "completed":
        try:
            result = backend.get_script_results(run_id)
        except loomloom.LoomLoomError as exc:
            _handle_loomloom_poll_error(run_id, exc)
            return
        st.session_state["loomloom_poll_failure_count"] = 0
        st.session_state["loomloom_poll_retry_after"] = 0.0
        st.session_state["loomloom_poll_paused"] = False
        st.session_state["loomloom_script_candidates"] = result.candidates
        st.session_state["loomloom_candidate_errors"] = result.errors
        st.session_state["loomloom_selected_candidate"] = 0
        st.session_state["loomloom_run_id"] = ""
        st.rerun(scope="app")
        return
    if run.status in {"failed", "cancelled", "canceled"}:
        st.session_state["loomloom_run_error"] = run.first_error_message or run.status
        st.session_state["loomloom_run_id"] = ""
        st.session_state["loomloom_poll_paused"] = False
        st.rerun(scope="app")
        return

    st.session_state["loomloom_poll_failure_count"] = 0
    st.session_state["loomloom_poll_retry_after"] = 0.0
    st.info(
        tr("LoomLoom Run Progress").format(
            completed=run.completed_tasks,
            total=run.total_tasks,
        )
    )


def _render_loomloom_script_generation(params):
    st.caption(tr("LoomLoom Batch Script Generation Help"))
    effective_token = _effective_loomloom_api_token()
    if not effective_token:
        st.warning(tr("Shengsuan Cloud API Key Required"))

    candidate_col, duration_col = st.columns(2)
    candidate_count = candidate_col.number_input(
        tr("Script Candidate Count"),
        min_value=1,
        max_value=loomloom.MAX_SCRIPT_CANDIDATES,
        step=1,
        key="loomloom_candidate_count",
    )
    duration_seconds = duration_col.number_input(
        tr("Target Script Duration Seconds"),
        min_value=10,
        max_value=600,
        step=10,
        key="loomloom_script_duration_seconds",
    )
    _set_runtime_config("ui", "loomloom_candidate_count", int(candidate_count))
    _set_runtime_config(
        "ui", "loomloom_script_duration_seconds", int(duration_seconds)
    )
    input_signature = _loomloom_script_signature(
        subject=params.video_subject,
        language=params.video_language,
        candidate_count=candidate_count,
        duration_seconds=duration_seconds,
        style=params.video_script_prompt,
        credential_fingerprint=(
            hashlib.sha256(effective_token.encode("utf-8")).hexdigest()
            if effective_token
            else ""
        ),
    )

    if st.button(
        tr("Get LoomLoom Quote"),
        key="loomloom_quote_scripts",
        use_container_width=True,
        type="secondary",
        icon=":material/request_quote:",
        disabled=not effective_token or bool(st.session_state.get("loomloom_run_id")),
    ):
        if not params.video_subject:
            st.toast(tr("Please Enter the Video Subject First"))
            st.warning(tr("Please Enter the Video Subject First"))
        else:
            try:
                backend = _create_loomloom_script_backend()
                batch = backend.prepare_script_batch(
                    subject=params.video_subject,
                    candidate_count=int(candidate_count),
                    language=params.video_language,
                    duration_seconds=int(duration_seconds),
                    style=params.video_script_prompt,
                )
                quote_result = backend.quote(batch)
            except (loomloom.LoomLoomError, ValueError) as exc:
                logger.warning(f"failed to quote LoomLoom scripts: error={exc}")
                st.error(str(exc))
            else:
                st.session_state["loomloom_script_batch"] = batch
                st.session_state["loomloom_script_quote"] = quote_result
                st.session_state["loomloom_script_input_signature"] = input_signature
                st.session_state["loomloom_client_request_id"] = f"mpt-{uuid4()}"
                st.session_state["loomloom_run_id"] = ""
                st.session_state["loomloom_run_status"] = "quoted"
                st.session_state["loomloom_run_error"] = ""
                st.session_state["loomloom_poll_failure_count"] = 0
                st.session_state["loomloom_poll_retry_after"] = 0.0
                st.session_state["loomloom_poll_paused"] = False
                st.session_state["loomloom_script_candidates"] = ()
                st.session_state["loomloom_candidate_errors"] = ()
                st.session_state["loomloom_confirm_charge"] = False
                logger.info(
                    "LoomLoom script quote ready: "
                    f"tasks={quote_result.task_count}, currency={quote_result.currency}, "
                    f"estimated_payable_t={quote_result.estimated_buyer_payable_t}"
                )

    quote_result = st.session_state.get("loomloom_script_quote")
    batch = st.session_state.get("loomloom_script_batch")
    if quote_result is not None and batch is not None:
        display_amount = (
            quote_result.estimated_buyer_payable_amount
            or f"{quote_result.estimated_buyer_payable_t} T"
        )
        st.success(
            tr(
                "LoomLoom Quote Summary Singular"
                if quote_result.task_count == 1
                else "LoomLoom Quote Summary"
            ).format(
                tasks=quote_result.task_count,
                amount=display_amount,
                currency=quote_result.currency,
            )
        )
        quote_is_current = (
            st.session_state.get("loomloom_script_input_signature") == input_signature
        )
        if not quote_is_current:
            st.warning(tr("LoomLoom Quote Changed Warning"))
        confirm_charge = st.checkbox(
            tr("Confirm LoomLoom Charge"),
            key="loomloom_confirm_charge",
            disabled=not quote_is_current,
        )
        run_in_progress = bool(st.session_state.get("loomloom_run_id"))
        if st.button(
            tr("Run LoomLoom Batch"),
            key="loomloom_execute_scripts",
            use_container_width=True,
            type="primary",
            disabled=(not quote_is_current or not confirm_charge or run_in_progress),
        ):
            try:
                execution = _create_loomloom_script_backend().execute(
                    batch,
                    client_request_id=st.session_state["loomloom_client_request_id"],
                    listing_version_id=quote_result.listing_version_id,
                    confirm=True,
                )
            except (loomloom.LoomLoomError, ValueError) as exc:
                logger.warning(f"failed to execute LoomLoom scripts: error={exc}")
                st.error(str(exc))
            else:
                st.session_state["loomloom_run_id"] = execution.run_id
                st.session_state["loomloom_run_status"] = "running"
                st.session_state["loomloom_poll_paused"] = False
                # Each quote allows starting only one billable batch. Background status relies solely on run_id;
                # quote and idempotent request ID can be discarded after submission. On failure, user must re-quote.
                st.session_state["loomloom_script_batch"] = None
                st.session_state["loomloom_script_quote"] = None
                st.session_state["loomloom_script_input_signature"] = ""
                st.session_state["loomloom_client_request_id"] = ""
                logger.info(
                    f"LoomLoom script run submitted: run_id={execution.run_id}, "
                    f"tasks={len(batch.input_rows)}"
                )
                st.toast(tr("LoomLoom Run Submitted"))

    run_error = str(st.session_state.get("loomloom_run_error", "") or "").strip()
    if run_error:
        st.error(tr("LoomLoom Run Failed").format(error=run_error))
    run_id = str(st.session_state.get("loomloom_run_id", "") or "").strip()
    if run_id and st.session_state.get("loomloom_poll_paused", False):
        retry_col, stop_col = st.columns(2)
        if retry_col.button(
            tr("Resume LoomLoom Status Check"),
            key="loomloom_resume_status_check",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state["loomloom_run_error"] = ""
            st.session_state["loomloom_poll_failure_count"] = 0
            st.session_state["loomloom_poll_retry_after"] = 0.0
            st.session_state["loomloom_poll_paused"] = False
            st.rerun(scope="app")
        if stop_col.button(
            tr("Stop Tracking LoomLoom Run"),
            key="loomloom_stop_tracking_run",
            use_container_width=True,
            type="secondary",
            help=tr("Stop Tracking LoomLoom Run Help"),
        ):
            # Only halts local status queries without claiming remote cancellation. Clear run_id only after user
            # confirms abandonment; subsequent billable runs require re-quoting and confirmation.
            st.session_state["loomloom_run_id"] = ""
            st.session_state["loomloom_run_error"] = ""
            st.session_state["loomloom_poll_paused"] = False
            st.rerun(scope="app")
    # Start 2-second polling only during active batches. Quoting and result phases do not create periodic fragments,
    # avoiding pointless network requests and reruns while the user stays on the page.
    if run_id and not st.session_state.get("loomloom_poll_paused", False):
        _render_loomloom_run_progress()
    _render_loomloom_candidates()


def _render_script_settings(panel, params):
    """Render script settings and update generation parameters."""
    with panel:
        with st.container(border=True):
            st.write(tr("Video Script Settings"))
            # The label row needs to host the 'Configure LLM' link, precluding text_area's built-in label.
            # Enclosing label and input in a shared field container overrides internal margins while
            # keeping exterior spacing consistent with other form controls.
            with st.container(key="video_subject_field"):
                with st.container(
                    key="video_subject_label_row",
                    horizontal=True,
                    vertical_alignment="center",
                    gap="small",
                ):
                    st.markdown(
                        tr("Video Subject"),
                        help=tr("Video Subject Help"),
                        width="content",
                    )
                    st.button(
                        tr("Configure LLM"),
                        key="open_llm_settings_from_subject",
                        type="tertiary",
                        on_click=_open_settings_dialog,
                        args=("llm",),
                    )
                params.video_subject = st.text_area(
                    tr("Video Subject"),
                    placeholder=tr("Video Subject Placeholder"),
                    height=96,
                    key="video_subject",
                    label_visibility="collapsed",
                ).strip()

            video_languages = [
                (tr("Auto Detect"), ""),
            ]
            for code in support_locales:
                video_languages.append((code, code))

            selected_language_code = stable_selectbox(
                tr("Script Language"),
                options=[value for _, value in video_languages],
                default_value=_saved_ui_choice(
                    "video_language",
                    [value for _, value in video_languages],
                    "",
                ),
                key="script_language_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_languages
                )[value],
            )
            params.video_language = selected_language_code
            _set_runtime_config("ui", "video_language", params.video_language)

            # Scope expander styles using keyed local containers, preserving native expander interactions
            # without inadvertently styling other expanders such as 'Basic Settings' at the top.
            with st.container(key="advanced_settings_script"):
                with st.expander(tr("Advanced Script Settings"), expanded=False):
                    script_backend_options = ["local", "loomloom"]
                    script_backend_labels = {
                        "local": tr("Local LLM Script Generation"),
                        "loomloom": tr("Shengsuan Cloud Batch Script Generation"),
                    }
                    script_backend_widget_key = localized_widget_key(
                        "script_generation_backend_select"
                    )
                    current_script_backend = st.session_state.get(
                        script_backend_widget_key,
                        _effective_script_generation_backend(),
                    )
                    script_generation_backend = stable_selectbox(
                        tr("Script Generation Method"),
                        options=script_backend_options,
                        default_value=_effective_script_generation_backend(),
                        key="script_generation_backend_select",
                        format_func=lambda value: script_backend_labels[value],
                        help=_script_generation_method_help(current_script_backend),
                    )
                    _set_runtime_config(
                        "app", "script_generation_backend", script_generation_backend
                    )

                    params.paragraph_number = st.slider(
                        tr("Script Paragraph Number"),
                        min_value=llm.MIN_SCRIPT_PARAGRAPH_NUMBER,
                        max_value=llm.MAX_SCRIPT_PARAGRAPH_NUMBER,
                        key="paragraph_number_input",
                    )
                    _set_runtime_config(
                        "ui", "paragraph_number", params.paragraph_number
                    )
                    params.video_script_prompt = st.text_area(
                        tr("Custom Script Requirements"),
                        height=100,
                        max_chars=llm.MAX_SCRIPT_PROMPT_LENGTH,
                        placeholder=tr("Custom Script Requirements Placeholder"),
                        key="video_script_prompt",
                    ).strip()
                    _set_runtime_config(
                        "ui", "video_script_prompt", params.video_script_prompt
                    )

                    system_prompt = st.text_area(
                        tr("Custom System Prompt"),
                        height=240,
                        max_chars=llm.MAX_SCRIPT_SYSTEM_PROMPT_LENGTH,
                        key="custom_system_prompt",
                    ).strip()
                    # Defaults are maintained by the service layer. Although the UI displays the default prompt directly,
                    # it is only passed with tasks if modified by the user, preventing tasks from freezing outdated defaults.
                    params.custom_system_prompt = (
                        ""
                        if system_prompt == llm.DEFAULT_SCRIPT_SYSTEM_PROMPT.strip()
                        else system_prompt
                    )
                    _set_runtime_config(
                        "ui", "custom_system_prompt", params.custom_system_prompt
                    )

                    restore_prompt_col, preview_prompt_col = st.columns(2)
                    if restore_prompt_col.button(
                        tr("Restore Default System Prompt"),
                        key="restore_default_system_prompt",
                        icon=":material/restart_alt:",
                        on_click=reset_script_system_prompt,
                        use_container_width=True,
                    ):
                        st.toast(tr("Default System Prompt Restored"))
                    if preview_prompt_col.button(
                        tr("Preview Final Prompt"),
                        key="preview_final_script_prompt",
                        icon=":material/preview:",
                        use_container_width=True,
                    ):
                        render_script_prompt_preview(
                            llm.build_script_prompt(
                                video_subject=params.video_subject,
                                language=params.video_language,
                                paragraph_number=params.paragraph_number,
                                video_script_prompt=params.video_script_prompt,
                                custom_system_prompt=params.custom_system_prompt,
                            )
                        )

            # Model discovery enhances video materials only, without altering the user-selected script provider.
            if _effective_script_generation_backend() == "loomloom":
                _render_loomloom_script_generation(params)
            else:
                _render_local_script_generation(params)
            params.video_script = st.text_area(
                tr("Video Script"),
                help=tr("Video Script Help"),
                height=180,
                key="video_script",
            )
            if _effective_script_generation_backend() == "loomloom":
                st.caption(tr("LoomLoom Video Terms Reuse Help"))
            elif st.button(
                tr("Generate Video Keywords"),
                key="auto_generate_terms",
                use_container_width=True,
                type="secondary",
                icon=":material/auto_awesome:",
            ):
                if not params.video_script:
                    # Video keywords require extraction from script copy; prompt early and skip model invocation when empty.
                    st.toast(tr("Please Enter the Video Subject"))
                    st.warning(tr("Please Enter the Video Subject"))
                else:
                    with st.spinner(tr("Generating Video Keywords")):
                        terms = _run_llm_read_operation(
                            "generate_terms",
                            lambda app_config_snapshot: llm.generate_terms(
                                params.video_subject,
                                params.video_script,
                                amount=8 if params.match_materials_to_script else 5,
                                match_script_order=params.match_materials_to_script,
                                app_config=app_config_snapshot,
                            ),
                        )
                        if "Error: " in terms:
                            st.error(tr(terms))
                        else:
                            st.session_state["video_terms"] = ", ".join(terms)

            params.video_terms = st.text_area(
                tr("Video Keywords"),
                help=tr("Video Keywords Help"),
                key="video_terms",
            )


def _render_video_settings(panel, params):
    """Render video settings and return local materials selected for this run."""
    uploaded_files = []
    with panel:
        with st.container(border=True):
            st.write(tr("Video Settings"))
            video_concat_modes = [
                (tr("Sequential"), "sequential"),
                (tr("Random"), "random"),
            ]
            video_source_labels = {
                "pexels": tr("Pexels"),
                "pixabay": tr("Pixabay"),
                "coverr": tr("Coverr"),
                "wavespeed": tr("WaveSpeed AI Video"),
                "volcengine_seedance": tr("Volcano Engine Seedance"),
                "ofox": tr("OFox AI Video"),
                "metaso_minimax": tr("Metaso MiniMax H3"),
                "muapi": tr("MuAPI AI Video"),
                "loomloom": tr("Shengsuan Cloud AI Video"),
                "openai_image": tr("OpenAI Compatible Text-to-Image"),
                "local": tr("Local file"),
            }
            saved_video_source_name = str(
                config.app.get("video_source", "pexels") or "pexels"
            )
            params.video_source = grouped_selectbox(
                tr("Video Source"),
                groups=(
                    (tr("Stock Video"), VIDEO_SOURCE_GROUPS["stock_video"]),
                    (tr("AI Video"), VIDEO_SOURCE_GROUPS["ai_video"]),
                    (tr("AI Image"), VIDEO_SOURCE_GROUPS["ai_image"]),
                    (tr("Local Material"), VIDEO_SOURCE_GROUPS["local"]),
                ),
                default_value=saved_video_source_name,
                key="video_source_select",
                format_func=video_source_labels.get,
                settings_label=tr("Configure Material Sources"),
                on_settings=_open_material_settings_dialog,
            )
            _set_runtime_config("app", "video_source", params.video_source)

            loomloom_video_capability = None
            if params.video_source == "loomloom":
                # Read cache early so aspect ratio controls below are bounded by current profile.
                # Streamlit reruns upon first key input, loading once immediately here.
                loomloom_video_capability = _load_loomloom_video_capability(
                    _effective_loomloom_api_token()
                )

            if params.video_source == "wavespeed":
                st.caption(tr("WaveSpeed AI Video Help"))
            if params.video_source == "volcengine_seedance":
                st.caption(tr("Volcano Engine Seedance Help"))
            if params.video_source == "ofox":
                st.caption(f"[OfoxAI]({OFOX_REFERRAL_URL}) · {tr('OFox AI Video Help')}")
            if params.video_source == "metaso_minimax":
                st.caption(tr("Metaso MiniMax H3 Help"))
            if params.video_source == "muapi":
                st.caption(tr("MuAPI AI Video Help"))
            if params.video_source == "local":
                # Streamlit file type validation is case-sensitive; permit both upper and lower case extensions.
                local_file_types = sorted(
                    extension.removeprefix(".")
                    for extension in LOCAL_MATERIAL_EXTENSIONS
                )
                uploaded_files = st.file_uploader(
                    tr("Upload Local Files"),
                    type=local_file_types
                    + [file_type.upper() for file_type in local_file_types],
                    accept_multiple_files=True,
                    key="local_video_materials_uploader",
                )

            # Script order matching enforces narrative progression from keywords through final synthesis;
            # sequential concat is the only logical choice when enabled. Syncing widget values avoids displaying
            # 'Random Concat' while retaining previous choices to restore when disabled.
            sync_script_order_concat_mode()
            selected_concat_mode = stable_selectbox(
                tr("Video Concat Mode"),
                options=[value for _, value in video_concat_modes],
                default_value=_saved_ui_choice(
                    "video_concat_mode",
                    [value for _, value in video_concat_modes],
                    VideoConcatMode.random.value,
                ),
                key="video_concat_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_concat_modes
                )[value],
                disabled=bool(st.session_state.get("match_materials_to_script", False)),
            )
            params.video_concat_mode = VideoConcatMode(selected_concat_mode)

            params.match_materials_to_script = st.checkbox(
                tr("Match Materials to Script Order"),
                help=tr("Match Materials to Script Order Help"),
                key="match_materials_to_script",
                on_change=sync_script_order_concat_mode,
            )
            _set_runtime_config(
                "app",
                "match_materials_to_script",
                params.match_materials_to_script,
            )
            # Sequential concat is an enforced value when order matching is enabled and should not overwrite
            # user preferences when toggled off; toggling off restores previous random/sequential settings.
            if not params.match_materials_to_script:
                _set_runtime_config(
                    "ui", "video_concat_mode", params.video_concat_mode.value
                )

            # Video transition mode
            video_transition_modes = [
                (tr("None"), VideoTransitionMode.none.value),
                (tr("Shuffle"), VideoTransitionMode.shuffle.value),
                (tr("FadeIn"), VideoTransitionMode.fade_in.value),
                (tr("FadeOut"), VideoTransitionMode.fade_out.value),
                (tr("SlideIn"), VideoTransitionMode.slide_in.value),
                (tr("SlideOut"), VideoTransitionMode.slide_out.value),
                (tr("ZoomIn"), VideoTransitionMode.zoom_in.value),
                (tr("ZoomOut"), VideoTransitionMode.zoom_out.value),
            ]
            selected_transition_mode = stable_selectbox(
                tr("Video Transition Mode"),
                options=[value for _, value in video_transition_modes],
                default_value=_saved_ui_choice(
                    "video_transition_mode",
                    [value for _, value in video_transition_modes],
                    VideoTransitionMode.none.value,
                ),
                key="video_transition_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_transition_modes
                )[value],
            )
            params.video_transition_mode = VideoTransitionMode(selected_transition_mode)
            _set_runtime_config(
                "ui",
                "video_transition_mode",
                params.video_transition_mode.value,
            )

            video_aspect_ratios = [
                (tr("Portrait"), VideoAspect.portrait.value),
                (tr("Landscape"), VideoAspect.landscape.value),
            ]
            if loomloom_video_capability is not None:
                ratio_labels = {value: label for label, value in video_aspect_ratios}
                video_aspect_ratios = [
                    (ratio_labels[value], value)
                    for value in loomloom_video_capability.aspect_ratios
                ]
            # 99% of Coverr footage is 16:9 Landscape; default Portrait results in heavy black letterboxing.
            # Use source-specific widget keys to memorize aspect choices per source:
            # - First switch to coverr -> default Landscape (index=1)
            # - Other sources keep Portrait (index=0)
            # - User manual aspect changes are stored in session_state and respected on next return.
            default_aspect_index = 1 if params.video_source == "coverr" else 0
            video_aspect_values = [value for _, value in video_aspect_ratios]
            video_aspect_config_key = f"video_aspect_{params.video_source}"
            selected_aspect_ratio = stable_selectbox(
                tr("Video Ratio"),
                options=video_aspect_values,
                default_value=_saved_ui_choice(
                    video_aspect_config_key,
                    video_aspect_values,
                    video_aspect_ratios[default_aspect_index][1],
                ),
                key=f"video_aspect_for_{params.video_source}",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_aspect_ratios
                )[value],
            )
            params.video_aspect = VideoAspect(selected_aspect_ratio)
            _set_runtime_config(
                "ui", video_aspect_config_key, params.video_aspect.value
            )

            video_fit_modes = [
                (tr("Fill and Crop"), VideoFitMode.cover.value),
                (tr("Fit with Black Bars"), VideoFitMode.contain.value),
            ]
            selected_fit_mode = stable_selectbox(
                tr("Video Fit Mode"),
                options=[value for _, value in video_fit_modes],
                default_value=_saved_ui_choice(
                    "video_fit_mode",
                    [value for _, value in video_fit_modes],
                    VideoFitMode.cover.value,
                ),
                key="video_fit_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_fit_modes
                )[value],
                help=tr("Video Fit Mode Help"),
            )
            params.video_fit_mode = VideoFitMode(selected_fit_mode)
            _set_runtime_config(
                "ui", "video_fit_mode", params.video_fit_mode.value
            )

            if params.video_source in {"local", "openai_image"}:
                image_motion_labels = {
                    ImageMotion.static.value: tr("Static Image"),
                    ImageMotion.zoom.value: tr("Slow Zoom"),
                }
                params.image_motion = ImageMotion(
                    stable_selectbox(
                        tr("Image Motion"),
                        options=list(image_motion_labels),
                        default_value=_saved_ui_choice(
                            "image_motion", list(image_motion_labels), ImageMotion.zoom.value,
                            # Switching language or material source recreates widgets while tasks hold the lock;
                            # restore the latest pending selection rather than writing back stale configuration.
                            section=config.snapshot_config_with_pending(config.ui),
                        ),
                        key="image_motion_select",
                        format_func=image_motion_labels.get,
                        help=tr("Image Motion Help"),
                    )
                )
                _set_runtime_config("ui", "image_motion", params.image_motion.value)

            # Remote duration range for MiniMax H3 is 4-15 seconds. Selecting MetaSo uses the full capability
            # range, preventing 2-3s from being billed as 4s, while matching WebUI with CLI and service layers.
            if params.video_source == "metaso_minimax":
                video_clip_durations = list(
                    range(
                        metaso_minimax.DEFAULT_MIN_DURATION_SECONDS,
                        metaso_minimax.DEFAULT_MAX_DURATION_SECONDS + 1,
                    )
                )
            elif params.video_source == "muapi":
                video_clip_durations = list(
                    range(
                        muapi.DEFAULT_MIN_DURATION_SECONDS,
                        muapi.DEFAULT_MAX_DURATION_SECONDS + 1,
                    )
                )
            else:
                video_clip_durations = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            params.video_clip_duration = stable_selectbox(
                tr("Clip Duration"),
                options=video_clip_durations,
                default_value=_saved_ui_choice(
                    "video_clip_duration",
                    video_clip_durations,
                    5
                    if params.video_source in {"metaso_minimax", "muapi"}
                    else 3,
                ),
                key="video_clip_duration_select",
                help=tr("Clip Duration Help"),
            )
            _set_runtime_config(
                "ui", "video_clip_duration", params.video_clip_duration
            )
            clip_speed_key = localized_widget_key("video_clip_speed_slider")
            # session_state may originate from legacy tasks, API parameters, or older page states.
            # Normalize before widget instantiation to keep valid choices while ensuring sliders receive finite floats in [0.5, 2.0].
            st.session_state[clip_speed_key] = utils.normalize_clip_speed(
                st.session_state.get(
                    clip_speed_key,
                    _saved_ui_number("video_clip_speed", 1.0, 0.5, 2.0),
                )
            )
            params.video_clip_speed = st.slider(
                tr("Clip Speed"),
                min_value=0.5,
                max_value=2.0,
                step=0.05,
                format="%.2fx",
                key=clip_speed_key,
                help=tr("Clip Speed Help"),
            )
            _set_runtime_config("ui", "video_clip_speed", params.video_clip_speed)
            video_count_options = [1, 2, 3, 4, 5]
            params.video_count = stable_selectbox(
                tr("Number of Videos Generated Simultaneously"),
                options=video_count_options,
                default_value=_saved_ui_choice(
                    "video_count", video_count_options, 1
                ),
                key="video_count_select",
            )
            _set_runtime_config("ui", "video_count", params.video_count)

            video_codec_options = [
                (tr("Default Video Encoder"), DEFAULT_VIDEO_CODEC_OPTION),
                ("libx264 (CPU)", "libx264"),
                ("NVIDIA NVENC (h264_nvenc)", "h264_nvenc"),
                ("AMD AMF (h264_amf)", "h264_amf"),
                ("Intel QSV (h264_qsv)", "h264_qsv"),
                ("Windows MediaFoundation (h264_mf)", "h264_mf"),
                ("macOS VideoToolbox (h264_videotoolbox)", "h264_videotoolbox"),
            ]
            saved_video_codec = config.app.get(
                "video_codec", DEFAULT_VIDEO_CODEC_OPTION
            )
            saved_video_codec_values = [item[1] for item in video_codec_options]
            if saved_video_codec not in saved_video_codec_values:
                # Legacy versions or manual editing may leave invalid values. Revert UI to 'Default' rather than pinning
                # an encoder for the user; the backend still resolves to libx264 via its stable policy.
                saved_video_codec = DEFAULT_VIDEO_CODEC_OPTION
            selected_video_codec = stable_selectbox(
                tr("Video Encoder"),
                options=saved_video_codec_values,
                default_value=saved_video_codec,
                key="video_encoder_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_codec_options
                )[value],
                help=tr("Video Encoder Help"),
            )
            if selected_video_codec == DEFAULT_VIDEO_CODEC_OPTION:
                # Default mode avoids persisting a concrete encoder, expressing 'follow project default'.
                _delete_runtime_config("app", "video_codec")
            else:
                _set_runtime_config("app", "video_codec", selected_video_codec)

            concurrency_options = [1, 2, 4, 6, 8]
            if params.video_source in {"pexels", "pixabay", "coverr"}:
                selected_material_concurrency = stable_selectbox(
                    tr("Material Concurrency"),
                    options=concurrency_options,
                    default_value=_saved_ui_choice(
                        "material_concurrency", concurrency_options, 1, config.app
                    ),
                    key="material_concurrency_select",
                    help=tr("Material Concurrency Help"),
                )
                _set_runtime_config(
                    "app", "material_concurrency", selected_material_concurrency
                )

            selected_clip_concurrency = stable_selectbox(
                tr("Clip Rendering Concurrency"),
                options=concurrency_options,
                default_value=_saved_ui_choice(
                    "video_clip_concurrency", concurrency_options, 1, config.app
                ),
                key="clip_rendering_concurrency_select",
                help=tr("Clip Rendering Concurrency Help"),
            )
            _set_runtime_config(
                "app", "video_clip_concurrency", selected_clip_concurrency
            )

            if params.video_source == "loomloom":
                _render_loomloom_video_settings(params)

            if params.video_source == "wavespeed":
                _render_wavespeed_video_settings(params)
            if params.video_source == "volcengine_seedance":
                _render_seedance_video_settings(params)
            if params.video_source == "ofox":
                _render_ofox_video_settings(params)
            if params.video_source == "metaso_minimax":
                _render_metaso_minimax_video_settings(params)
            if params.video_source == "muapi":
                _render_muapi_video_settings(params)
    return uploaded_files


def _render_wavespeed_video_settings(params):
    """
    Render WaveSpeed generation quantity estimates and billing confirmation.

    Billing is per video clip; users must see an estimate of clip counts before submitting.
    Estimation runs entirely locally: dividing estimated voiceover duration range by clip duration.
    Since material pipelines generate clips incrementally and stop once required duration is satisfied,
    actual clip count is governed by runtime; estimation serves as a magnitude guide without participating in task execution.
    """
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    estimated_range = _estimate_voiceover_duration_range(
        str(params.video_script or ""),
        params.voice_rate,
    )
    if estimated_range:
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("WaveSpeed Billing Notice").format(min=min_clips, max=max_clips)
        )
    else:
        st.warning(tr("WaveSpeed Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm WaveSpeed Charge"),
        key="wavespeed_confirm_charge",
        help=tr("Confirm WaveSpeed Charge Help"),
    )


def _render_seedance_video_settings(params):
    """Display estimated billable task count and require explicit user confirmation of Volcengine generation costs."""
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    estimated_range = _estimate_voiceover_duration_range(
        str(params.video_script or ""), params.voice_rate
    )
    if estimated_range:
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("Volcano Engine Seedance Billing Notice").format(
                min=min_clips, max=max_clips
            )
        )
    else:
        st.warning(tr("Volcano Engine Seedance Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm Volcano Engine Seedance Charge"),
        key="volcengine_seedance_confirm_charge",
        help=tr("Confirm Volcano Engine Seedance Charge Help"),
    )


def _render_ofox_video_settings(params):
    """Display estimated billable task count and require explicit user confirmation of OFox generation costs."""
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    estimated_range = _estimate_voiceover_duration_range(
        str(params.video_script or ""), params.voice_rate
    )
    if estimated_range:
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("OFox Billing Notice").format(min=min_clips, max=max_clips)
        )
    else:
        st.warning(tr("OFox Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm OFox Charge"),
        key="ofox_confirm_charge",
        help=tr("Confirm OFox Charge Help"),
    )


def _render_metaso_minimax_video_settings(params):
    """Display estimated billable task count and require explicit user confirmation of MetaSo MiniMax generation costs."""
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    voice_mode = st.session_state.get(
        localized_widget_key("voice_mode_control"),
        config.ui.get("voice_mode"),
    )
    if voice_mode == VOICE_MODE_UPLOAD:
        # Video settings render before audio settings, making it impossible to reliably read the actual duration
        # of freshly uploaded audio files. Upload mode skips text-based estimates to prevent users thinking
        # a 5-second audio creates multiple billable tasks; runtime uses actual file duration.
        st.warning(
            tr("Metaso MiniMax Billing Notice Uploaded Audio").format(
                resolution=str(
                    config.app.get(
                        "metaso_minimax_resolution",
                        metaso_minimax.DEFAULT_RESOLUTION,
                    )
                    or metaso_minimax.DEFAULT_RESOLUTION
                ),
                duration=clip_duration,
                count=video_count,
            )
        )
    elif estimated_range := _estimate_voiceover_duration_range(
        str(params.video_script or ""), params.voice_rate
    ):
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("Metaso MiniMax Billing Notice").format(
                min=min_clips,
                max=max_clips,
                resolution=str(
                    config.app.get(
                        "metaso_minimax_resolution",
                        metaso_minimax.DEFAULT_RESOLUTION,
                    )
                    or metaso_minimax.DEFAULT_RESOLUTION
                ),
            )
        )
    else:
        st.warning(tr("Metaso MiniMax Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm Metaso MiniMax Charge"),
        key="metaso_minimax_confirm_charge",
        help=tr("Confirm Metaso MiniMax Charge Help"),
    )


def _render_muapi_video_settings(params):
    """Show an estimated MuAPI task count and require explicit billing consent."""
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    if estimated_range := _estimate_voiceover_duration_range(
        str(params.video_script or ""), params.voice_rate
    ):
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("MuAPI Billing Notice").format(min=min_clips, max=max_clips)
        )
    else:
        st.warning(tr("MuAPI Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm MuAPI Charge"),
        key="muapi_confirm_charge",
        help=tr("Confirm MuAPI Charge Help"),
    )


def _estimate_voiceover_duration_range(
    text: str, voice_rate: float
) -> tuple[float, float] | None:
    """
    Locally estimate full narration duration, returning conservative lower and upper bounds in seconds.

    This estimate helps users gauge script volume before calling paid TTS, without participating in task execution.
    Chinese, Japanese, and Korean are estimated by character speed, while space-delimited languages use word speed,
    factoring in punctuation pauses. Differences across providers, voices, and cadence cause variance,
    so the UI displays a range rather than a misleading single value.
    """
    normalized_text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized_text:
        return None

    script_chars = re.findall(
        r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]",
        normalized_text,
    )
    remaining_text = re.sub(
        r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]",
        " ",
        normalized_text,
    )
    words = re.findall(r"\b[\w]+(?:[-'’][\w]+)*\b", remaining_text, re.UNICODE)
    punctuation_count = len(re.findall(r"[,，.。!?！？;；:：]", normalized_text))

    # 4.2 chars/sec and 2.6 words/sec approximate conversational narration speed; punctuation adds ~0.12s pause.
    # voice_rate acts as an adjustment factor. Some generative TTS models do not strictly adhere to multipliers,
    # so a +/-15% range is preserved to avoid implying exact server equivalence.
    base_seconds = len(script_chars) / 4.2 + len(words) / 2.6 + punctuation_count * 0.12
    if base_seconds <= 0:
        return None

    normalized_rate = max(float(voice_rate or 1.0), 0.1)
    estimated_seconds = base_seconds / normalized_rate
    return (
        round(max(estimated_seconds * 0.85, 1.0), 1),
        round(max(estimated_seconds * 1.15, 1.0), 1),
    )


def _get_voice_preview_sample(voice_name: str) -> str:
    """Return short preview text suitable for the current voice, rather than the user's full video script."""
    # When ElevenLabs voices lack explicit language tags, detect Vietnamese characters in the display name
    # to select preview text, avoiding judging voice quality with a mismatched language.
    if voice.is_elevenlabs_voice(voice_name):
        parts = voice_name.split(":", 2)
        display = parts[2] if len(parts) >= 3 else ""
        vietnamese_chars = set("àáâãèéêìíòóôõùúýăđơưÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ")
        if any(char in vietnamese_chars for char in display):
            return "Xin chào, đây là đoạn âm thanh thử nghiệm giọng nói."
    return tr("Voice Example")


def _voice_preview_fingerprint(
    *,
    preview_type: str,
    content: str,
    tts_server: str,
    voice_name: str,
    voice_rate: float,
    voice_volume: float,
    provider_signature: dict,
) -> str:
    """Generate audio preview cache fingerprint; invalidates old previews when any voice parameter changes."""
    payload = {
        "preview_type": preview_type,
        "content": content,
        "tts_server": tts_server,
        "voice_name": voice_name,
        "voice_rate": voice_rate,
        "voice_volume": voice_volume,
        "provider_signature": provider_signature,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _credential_signature(value: str) -> str:
    """
    Generate credential signature used solely for cache invalidation.

    Signatures are never written to config, logs, or task files. Changing API keys alters the signature,
    forcing fresh calls to the voice service to prevent stale caches making invalid credentials look functional.
    """
    normalized_value = str(value or "")
    if not normalized_value:
        return ""
    return hashlib.sha256(normalized_value.encode("utf-8")).hexdigest()


def _get_voxcpm_reference_audio() -> bytes | None:
    """Return the current session's normalized reference audio, if any."""
    payload = st.session_state.get(VOXCPM_REFERENCE_AUDIO_SESSION_KEY)
    if not isinstance(payload, dict):
        return None
    audio_bytes = payload.get("audio_bytes")
    return bytes(audio_bytes) if isinstance(audio_bytes, bytes) else None


def _get_voxcpm_reference_audio_digest() -> str:
    payload = st.session_state.get(VOXCPM_REFERENCE_AUDIO_SESSION_KEY)
    if not isinstance(payload, dict):
        return ""
    digest = payload.get("audio_digest")
    return str(digest) if isinstance(digest, str) else ""


def _get_voxcpm_prompt_audio() -> bytes | None:
    payload = st.session_state.get(VOXCPM_PROMPT_AUDIO_SESSION_KEY)
    if not isinstance(payload, dict):
        return None
    audio_bytes = payload.get("audio_bytes")
    return bytes(audio_bytes) if isinstance(audio_bytes, bytes) else None


def _get_voxcpm_prompt_audio_digest() -> str:
    payload = st.session_state.get(VOXCPM_PROMPT_AUDIO_SESSION_KEY)
    if not isinstance(payload, dict):
        return ""
    digest = payload.get("audio_digest")
    return str(digest) if isinstance(digest, str) else ""


def _get_voxcpm_prompt_text() -> str:
    return str(st.session_state.get(VOXCPM_PROMPT_TEXT_SESSION_KEY, "") or "").strip()


def _clear_voxcpm_separate_prompt_audio() -> None:
    st.session_state.pop("voxcpm_prompt_audio_uploader", None)
    st.session_state.pop(VOXCPM_PROMPT_AUDIO_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY, None)


def _clear_voxcpm_prompt_state() -> None:
    _clear_voxcpm_separate_prompt_audio()
    _clear_voxcpm_prompt_transcript()
    st.session_state.pop(VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_PROMPT_EXAMPLE_MODE_SESSION_KEY, None)


def _clear_voxcpm_prompt_transcript() -> None:
    st.session_state.pop(VOXCPM_PROMPT_TEXT_SESSION_KEY, None)


def _sync_voxcpm_prompt_example_mode(use_separate_prompt_audio: bool) -> None:
    """Invalidate the transcript whenever its effective example changes mode."""
    previous_mode = st.session_state.get(VOXCPM_PROMPT_EXAMPLE_MODE_SESSION_KEY)
    current_mode = bool(use_separate_prompt_audio)
    if previous_mode is not None and bool(previous_mode) != current_mode:
        _clear_voxcpm_prompt_transcript()
    st.session_state[VOXCPM_PROMPT_EXAMPLE_MODE_SESSION_KEY] = current_mode


def _get_voxcpm_effective_prompt_audio() -> bytes | None:
    if not st.session_state.get(VOXCPM_HIGH_FIDELITY_SESSION_KEY, False):
        return None
    if st.session_state.get(VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY, False):
        return _get_voxcpm_prompt_audio()
    return _get_voxcpm_reference_audio()


def _get_voxcpm_effective_prompt_audio_digest() -> str:
    if not st.session_state.get(VOXCPM_HIGH_FIDELITY_SESSION_KEY, False):
        return ""
    if st.session_state.get(VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY, False):
        return _get_voxcpm_prompt_audio_digest()
    return _get_voxcpm_reference_audio_digest()


def _get_voxcpm_prompt_validation_error() -> str:
    if not st.session_state.get(VOXCPM_HIGH_FIDELITY_SESSION_KEY, False):
        return ""
    upload_error = st.session_state.get(VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY)
    if upload_error:
        return str(upload_error)
    prompt_text = _get_voxcpm_prompt_text()
    if not prompt_text:
        return tr("VoxCPM Prompt Text Required")
    if (
        st.session_state.get(VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY, False)
        and not _get_voxcpm_prompt_audio()
    ):
        return tr("VoxCPM Prompt Pair Required")
    return ""


def _get_voxcpm_preview_validation_error(
    selected_tts_server: str,
    voice_name: str,
) -> str:
    if selected_tts_server != "voxcpm" and not voice.is_voxcpm_voice(voice_name):
        return ""
    reference_error = st.session_state.get(VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY)
    if reference_error:
        return str(reference_error)
    return _get_voxcpm_prompt_validation_error()


def _sync_voxcpm_reference_audio(uploaded_file) -> bytes | None:
    """Normalize a new upload once and keep only bounded WAV bytes in session."""
    if uploaded_file is None:
        st.session_state.pop(VOXCPM_REFERENCE_AUDIO_SESSION_KEY, None)
        st.session_state.pop(VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY, None)
        _clear_voxcpm_prompt_state()
        st.session_state.pop(VOXCPM_HIGH_FIDELITY_SESSION_KEY, None)
        return None

    upload_size = int(getattr(uploaded_file, "size", 0) or 0)
    if upload_size <= 0:
        st.session_state.pop(VOXCPM_REFERENCE_AUDIO_SESSION_KEY, None)
        _clear_voxcpm_prompt_state()
        st.session_state[VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY] = tr(
            "VoxCPM Reference Audio Empty"
        )
        return None
    if upload_size > voice.VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES:
        st.session_state.pop(VOXCPM_REFERENCE_AUDIO_SESSION_KEY, None)
        _clear_voxcpm_prompt_state()
        st.session_state[VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY] = tr(
            "VoxCPM Reference Audio Upload Too Large"
        ).format(limit=voice.VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES // (1024 * 1024))
        return None

    previous_digest = _get_voxcpm_reference_audio_digest()
    raw_audio = uploaded_file.getvalue()
    upload_digest = hashlib.sha256(raw_audio).hexdigest()
    cached = st.session_state.get(VOXCPM_REFERENCE_AUDIO_SESSION_KEY)
    if isinstance(cached, dict) and cached.get("upload_digest") == upload_digest:
        st.session_state.pop(VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY, None)
        return _get_voxcpm_reference_audio()

    try:
        with st.spinner(tr("Validating VoxCPM Reference Audio")):
            wav_audio = voice.prepare_voxcpm_reference_audio(
                raw_audio,
                Path(str(getattr(uploaded_file, "name", ""))).suffix,
            )
    except ValueError as exc:
        # Do not retain a prior clip after the user selects a replacement that
        # fails validation. Otherwise preview and generation could silently use
        # a different person's voice than the one currently shown in the UI.
        st.session_state.pop(VOXCPM_REFERENCE_AUDIO_SESSION_KEY, None)
        _clear_voxcpm_prompt_state()
        st.session_state[VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY] = str(exc)
        return None

    st.session_state[VOXCPM_REFERENCE_AUDIO_SESSION_KEY] = {
        "upload_digest": upload_digest,
        "audio_digest": hashlib.sha256(wav_audio).hexdigest(),
        "audio_bytes": wav_audio,
    }
    if (
        previous_digest != hashlib.sha256(wav_audio).hexdigest()
        and not st.session_state.get(VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY, False)
    ):
        _clear_voxcpm_prompt_transcript()
    st.session_state.pop(VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY, None)
    return bytes(wav_audio)


def _sync_voxcpm_prompt_audio(uploaded_file) -> bytes | None:
    """Normalize the optional delivery example without persisting its content."""
    if uploaded_file is None:
        st.session_state.pop(VOXCPM_PROMPT_AUDIO_SESSION_KEY, None)
        st.session_state.pop(VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY, None)
        return None

    upload_size = int(getattr(uploaded_file, "size", 0) or 0)
    if upload_size <= 0:
        st.session_state.pop(VOXCPM_PROMPT_AUDIO_SESSION_KEY, None)
        st.session_state[VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY] = tr(
            "VoxCPM Prompt Audio Empty"
        )
        return None
    if upload_size > voice.VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES:
        st.session_state.pop(VOXCPM_PROMPT_AUDIO_SESSION_KEY, None)
        st.session_state[VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY] = tr(
            "VoxCPM Prompt Audio Upload Too Large"
        ).format(limit=voice.VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES // (1024 * 1024))
        return None

    previous_digest = _get_voxcpm_prompt_audio_digest()
    raw_audio = uploaded_file.getvalue()
    upload_digest = hashlib.sha256(raw_audio).hexdigest()
    cached = st.session_state.get(VOXCPM_PROMPT_AUDIO_SESSION_KEY)
    if isinstance(cached, dict) and cached.get("upload_digest") == upload_digest:
        st.session_state.pop(VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY, None)
        return _get_voxcpm_prompt_audio()

    try:
        with st.spinner(tr("Validating VoxCPM Prompt Audio")):
            wav_audio = voice.prepare_voxcpm_reference_audio(
                raw_audio,
                Path(str(getattr(uploaded_file, "name", ""))).suffix,
            )
    except ValueError as exc:
        st.session_state.pop(VOXCPM_PROMPT_AUDIO_SESSION_KEY, None)
        st.session_state[VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY] = str(exc)
        return None

    st.session_state[VOXCPM_PROMPT_AUDIO_SESSION_KEY] = {
        "upload_digest": upload_digest,
        "audio_digest": hashlib.sha256(wav_audio).hexdigest(),
        "audio_bytes": wav_audio,
    }
    if previous_digest != hashlib.sha256(wav_audio).hexdigest():
        _clear_voxcpm_prompt_transcript()
    st.session_state.pop(VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY, None)
    return bytes(wav_audio)


def _get_voice_preview_provider_signature(tts_server: str) -> dict:
    """
    Return non-sensitive Provider configurations that affect voice preview outcomes.

    API keys only participate in cache fingerprints as one-way digests; raw credentials never enter cache or logs.
    Re-generate previews whenever model, service URL, region, or credentials change, preventing playback of audio
    under stale provider settings that could mislead users into thinking current settings are active.
    """
    if tts_server == "azure-tts-v2":
        return {
            "speech_region": config.azure.get("speech_region", ""),
            "credential": _credential_signature(config.azure.get("speech_key", "")),
        }
    if tts_server == "siliconflow":
        return {
            "credential": _credential_signature(config.siliconflow.get("api_key", ""))
        }
    if tts_server == "gemini-tts":
        return {
            "credential": _credential_signature(config.app.get("gemini_api_key", ""))
        }
    if tts_server == "mimo-tts":
        return {"credential": _credential_signature(config.app.get("mimo_api_key", ""))}
    if tts_server == "minimax-tts":
        return {
            "base_url": voice.get_minimax_tts_endpoint(),
            "model_id": config.minimax_tts.get("model_id", ""),
            "voice_id": config.minimax_tts.get("voice_id", ""),
            "credential": _credential_signature(voice.get_minimax_tts_api_key()),
        }
    if tts_server == "elevenlabs":
        return {
            "model_id": config.elevenlabs.get("model_id", ""),
            "credential": _credential_signature(config.elevenlabs.get("api_key", "")),
        }
    if tts_server == "chatterbox":
        return {
            "base_url": config.chatterbox.get("base_url", ""),
            "model_id": config.chatterbox.get("model_id", ""),
            "credential": _credential_signature(config.chatterbox.get("api_key", "")),
        }
    if tts_server == "kokoro":
        return {
            "base_url": config.kokoro.get("base_url", ""),
            "model_id": config.kokoro.get("model_id", ""),
            "credential": _credential_signature(config.kokoro.get("api_key", "")),
        }
    if tts_server == "voxcpm":
        return {
            "base_url": config.voxcpm.get("base_url", ""),
            "model_id": config.voxcpm.get("model_id", ""),
            "voice_id": config.voxcpm.get("voice_id", "default"),
            "credential": _credential_signature(config.voxcpm.get("api_key", "")),
            "reference_audio": _get_voxcpm_reference_audio_digest(),
            "prompt_audio": _get_voxcpm_effective_prompt_audio_digest(),
            "prompt_text": _credential_signature(
                _get_voxcpm_prompt_text()
                if st.session_state.get(VOXCPM_HIGH_FIDELITY_SESSION_KEY, False)
                else ""
            ),
        }
    return {}


def _synthesize_voice_preview(
    *,
    content: str,
    preview_type: str,
    selected_tts_server: str,
    voice_name: str,
    voice_rate: float,
    voice_volume: float,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
) -> dict | None:
    """Generate audio preview and store in in-memory cache; temporary files are not retained across sessions."""
    if selected_tts_server == "chatterbox":
        _sync_chatterbox_config_from_session_state()
    if selected_tts_server == "kokoro":
        _sync_kokoro_config_from_session_state()

    temp_dir = utils.storage_dir("temp", create=True)
    audio_file = os.path.join(temp_dir, f"tmp-voice-{str(uuid4())}.mp3")
    logger.info(
        f"generating {preview_type} voice preview: "
        f"voice={voice_name}, rate={voice_rate}, volume={voice_volume}, "
        f"text_length={len(content)}"
    )
    try:
        with config.try_runtime_config_lock() as lock_acquired:
            if not lock_acquired:
                return {"busy": True}
            tts_kwargs = {
                "text": content,
                "voice_name": voice_name,
                "voice_rate": voice_rate,
                "voice_file": audio_file,
                "voice_volume": voice_volume,
            }
            if voxcpm_reference_audio is not None:
                tts_kwargs["voxcpm_reference_audio"] = voxcpm_reference_audio
            if voxcpm_prompt_audio is not None:
                tts_kwargs["voxcpm_prompt_audio"] = voxcpm_prompt_audio
                tts_kwargs["voxcpm_prompt_text"] = voxcpm_prompt_text
            sub_maker = voice.tts(**tts_kwargs)
        if not sub_maker or not os.path.exists(audio_file):
            logger.error(f"{preview_type} voice preview did not produce an audio file")
            return None

        with open(audio_file, "rb") as file:
            audio_bytes = file.read()
        if not audio_bytes:
            logger.error(f"voice preview audio file is empty: {audio_file}")
            return None

        duration = voice.get_audio_duration(audio_file)
        if (
            not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            logger.warning(
                f"voice preview duration is unavailable: "
                f"preview_type={preview_type}, voice={voice_name}"
            )
            duration = None

        return {
            "audio_bytes": audio_bytes,
            "mime_type": _detect_audio_mime(audio_file, audio_bytes),
            "duration": duration,
            "preview_type": preview_type,
            "sub_maker": sub_maker,
            # Ensure the video panel preceding the audio panel uses full preview duration matching current settings exactly;
            # short previews or stale scripts must never alter recommended material counts.
            "content_digest": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "tts_server": selected_tts_server,
            "voice_name": voice_name,
            "voice_rate": float(voice_rate),
        }
    finally:
        # Browser player uses in-memory bytes; clean up file upon reading to prevent temp files accumulating from frequent previews.
        try:
            os.remove(audio_file)
        except FileNotFoundError:
            pass
        except OSError as exc:
            # Cleanup failure should not overwrite genuine TTS responses or exceptions, but preserve paths and system errors
            # to facilitate diagnosing environment issues such as permissions or read-only filesystems.
            logger.warning(
                f"failed to delete voice preview file {audio_file}: {str(exc)}"
            )


def _render_voice_preview(params, friendly_names, selected_tts_server, voice_name):
    """Render low-cost short previews, full copy duration estimates, and on-demand full voice previews."""
    if not friendly_names:
        return

    script_content = str(params.video_script or "").strip()
    estimated_range = _estimate_voiceover_duration_range(
        script_content,
        params.voice_rate,
    )
    if estimated_range:
        st.caption(
            tr("Estimated Voiceover Duration").format(
                min=estimated_range[0],
                max=estimated_range[1],
            )
        )
    else:
        st.caption(tr("Voiceover Script Required"))

    sample_content = _get_voice_preview_sample(voice_name)
    provider_signature = _get_voice_preview_provider_signature(selected_tts_server)
    preview_validation_error = _get_voxcpm_preview_validation_error(
        selected_tts_server,
        voice_name,
    )
    preview_columns = st.columns(2)
    short_preview_requested = preview_columns[0].button(
        tr("Play Voice"),
        key="play_voice_button",
        icon=":material/graphic_eq:",
        use_container_width=True,
        disabled=bool(preview_validation_error),
    )
    full_preview_requested = preview_columns[1].button(
        tr("Generate Full Voiceover Preview"),
        key="generate_full_voiceover_preview_button",
        icon=":material/article:",
        help=tr("Full Voiceover Preview Cost Hint"),
        use_container_width=True,
        disabled=not bool(script_content) or bool(preview_validation_error),
    )

    preview_type = ""
    preview_content = ""
    if short_preview_requested:
        preview_type = "sample"
        preview_content = sample_content
    elif full_preview_requested:
        preview_type = "full"
        preview_content = script_content

    sample_fingerprint = _voice_preview_fingerprint(
        preview_type="sample",
        content=sample_content,
        tts_server=selected_tts_server,
        voice_name=voice_name,
        voice_rate=params.voice_rate,
        voice_volume=params.voice_volume,
        provider_signature=provider_signature,
    )
    full_fingerprint = (
        _voice_preview_fingerprint(
            preview_type="full",
            content=script_content,
            tts_server=selected_tts_server,
            voice_name=voice_name,
            voice_rate=params.voice_rate,
            voice_volume=params.voice_volume,
            provider_signature=provider_signature,
        )
        if script_content
        else ""
    )

    if preview_type:
        requested_fingerprint = (
            sample_fingerprint if preview_type == "sample" else full_fingerprint
        )
        cached_preview = st.session_state.get("voice_preview_audio")
        if (
            not cached_preview
            or cached_preview.get("fingerprint") != requested_fingerprint
        ):
            try:
                with st.spinner(tr("Synthesizing Voice")):
                    preview_result = _synthesize_voice_preview(
                        content=preview_content,
                        preview_type=preview_type,
                        selected_tts_server=selected_tts_server,
                        voice_name=voice_name,
                        voice_rate=params.voice_rate,
                        voice_volume=params.voice_volume,
                        voxcpm_reference_audio=(
                            _get_voxcpm_reference_audio()
                            if selected_tts_server == "voxcpm"
                            else None
                        ),
                        voxcpm_prompt_audio=(
                            _get_voxcpm_effective_prompt_audio()
                            if selected_tts_server == "voxcpm"
                            else None
                        ),
                        voxcpm_prompt_text=(
                            _get_voxcpm_prompt_text()
                            if selected_tts_server == "voxcpm"
                            else ""
                        ),
                    )
            except Exception as exc:
                logger.exception(f"failed to generate {preview_type} voice preview")
                st.error(tr("Voice Preview Failed").format(error=str(exc)))
            else:
                if preview_result and preview_result.get("busy"):
                    st.warning(tr("Voice Preview Busy"))
                elif preview_result:
                    preview_result["fingerprint"] = requested_fingerprint
                    st.session_state["voice_preview_audio"] = preview_result
                    if (
                        preview_type == "full"
                        and params.video_source == "loomloom"
                        and isinstance(preview_result.get("duration"), (int, float))
                        and math.isfinite(preview_result["duration"])
                        and preview_result["duration"] > 0
                    ):
                        # Video settings render before audio settings. Trigger a rerun upon successful full preview
                        # so material count above re-recommends immediately based on actual narration duration.
                        st.session_state["loomloom_video_scene_autofill_digest"] = (
                            preview_result["content_digest"]
                        )
                        st.rerun()
                else:
                    st.error(tr("Voice Preview No Audio"))

    cached_preview = st.session_state.get("voice_preview_audio")
    valid_fingerprints = {sample_fingerprint, full_fingerprint}
    if (
        cached_preview
        and cached_preview.get("fingerprint") in valid_fingerprints
        and cached_preview.get("audio_bytes")
    ):
        # Autoplay only when the user explicitly clicks 'Preview Voice' in this run. Other Streamlit widgets
        # also trigger page reruns; permanently enabling autoplay on cached audio would replay old previews from scratch
        # whenever any setting is modified. Full voice preview retains manual playback to prevent longer audio from interrupting users.
        should_autoplay = bool(
            short_preview_requested
            and cached_preview.get("preview_type") == "sample"
            and cached_preview.get("fingerprint") == sample_fingerprint
        )
        st.audio(
            cached_preview["audio_bytes"],
            format=cached_preview.get("mime_type", "audio/mp3"),
            autoplay=should_autoplay,
        )
        if cached_preview.get("preview_type") == "full":
            duration = cached_preview.get("duration")
            if isinstance(duration, (int, float)) and duration > 0:
                st.caption(
                    tr("Actual Voiceover Duration").format(duration=f"{duration:.1f}")
                )
            else:
                st.warning(tr("Voice Preview Duration Unavailable"))


def _get_reusable_full_voice_preview(params, voice_mode: str) -> dict | None:
    """
    Return full voice preview cache that matches current generation parameters exactly.

    Only full copy previews are reused; short voice samples can never enter production tasks.
    Fingerprints cover script, provider, voice, rate, volume, and non-sensitive config digests;
    any parameter change naturally falls back to the standard TTS pipeline. Subtitle timelines and valid duration
    are also prerequisites to prevent Edge subtitle pipelines losing SubMaker when audio is reused alone.
    """
    if voice_mode != VOICE_MODE_TTS:
        return None

    script_content = str(params.video_script or "").strip()
    selected_tts_server = config.ui.get("tts_server", "azure-tts-v1")
    if (
        not script_content
        or not params.voice_name
        # Production videos apply voice volume during MoviePy synthesis; certain providers also apply gain directly during TTS.
        # Reusing preview audio at non-default volume may cause double amplification, so conservatively fall back
        # to the standard pipeline to avoid special-casing providers for niche scenarios.
        or not math.isclose(float(params.voice_volume), 1.0)
    ):
        return None

    expected_fingerprint = _voice_preview_fingerprint(
        preview_type="full",
        content=script_content,
        tts_server=selected_tts_server,
        voice_name=params.voice_name,
        voice_rate=params.voice_rate,
        voice_volume=params.voice_volume,
        provider_signature=_get_voice_preview_provider_signature(selected_tts_server),
    )
    cached_preview = st.session_state.get("voice_preview_audio")
    if (
        not cached_preview
        or cached_preview.get("fingerprint") != expected_fingerprint
        or cached_preview.get("preview_type") != "full"
        or not cached_preview.get("audio_bytes")
        or cached_preview.get("sub_maker") is None
    ):
        return None

    duration = cached_preview.get("duration")
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        return None

    return {
        "audio_bytes": bytes(cached_preview["audio_bytes"]),
        "duration": float(duration),
        "sub_maker": cached_preview["sub_maker"],
        "script": script_content,
        "voice_name": params.voice_name,
        "voice_rate": float(params.voice_rate),
        "voice_volume": float(params.voice_volume),
    }


def _sync_minimax_tts_api_key_input():
    """
    Synchronize MiniMax TTS password widget and return currently valid key.

    Reusing MiniMax LLM key is permitted when the dedicated TTS key is empty. Shared keys are used solely for current
    widgets and requests, not automatically duplicated into [minimax_tts], preventing duplicate credentials in config.
    """
    widget_key = "minimax_tts_api_key_input"
    configured_key = str(config.minimax_tts.get("api_key", "") or "").strip()
    shared_key = str(
        config.app.get("minimax_api_key", "") or os.getenv("MINIMAX_API_KEY", "") or ""
    ).strip()
    effective_key = configured_key or shared_key
    had_widget_state = widget_key in st.session_state
    entered_key = str(st.session_state.get(widget_key, "") or "").strip()

    if not entered_key and effective_key:
        # Browser reconnection may replay empty password states. Restore configured credentials to prevent blanks overwriting config
        # and ensure the current rerun preview request can directly use a valid key.
        st.session_state[widget_key] = effective_key
        entered_key = effective_key
        if had_widget_state:
            logger.debug("restored MiniMax TTS API key after empty session replay")
    elif not had_widget_state:
        st.session_state[widget_key] = effective_key
        entered_key = effective_key

    if entered_key and entered_key != effective_key:
        _set_runtime_config("minimax_tts", "api_key", entered_key)

    return entered_key


def _get_cached_minimax_voices(api_key: str, endpoint: str) -> list[dict[str, str]]:
    """Read MiniMax voice query results in current session by site and credential digest."""
    cache = st.session_state.get("minimax_tts_voice_catalog_cache", {})
    cache_key = f"{endpoint}|{_credential_signature(api_key)}"
    cached_voices = cache.get(cache_key, [])
    return cached_voices if isinstance(cached_voices, list) else []


def _cache_minimax_voices(
    api_key: str,
    endpoint: str,
    voices: list[dict[str, str]],
):
    """Cache actively queried voices to prevent repeated requests to MiniMax after regular widget reruns."""
    cache = st.session_state.setdefault("minimax_tts_voice_catalog_cache", {})
    cache_key = f"{endpoint}|{_credential_signature(api_key)}"
    cache[cache_key] = voices


def _render_minimax_tts_settings() -> tuple[list[str], dict[str, str]]:
    """Render MiniMax TTS configuration, returning options and copy used by unified voice selector."""
    effective_api_key = _sync_minimax_tts_api_key_input()
    effective_api_key = st.text_input(
        tr("MiniMax TTS API Key"),
        type="password",
        key="minimax_tts_api_key_input",
    ).strip()

    dedicated_key = str(config.minimax_tts.get("api_key", "") or "").strip()
    minimax_tts_endpoints = [voice.MINIMAX_TTS_GLOBAL_URL, voice.MINIMAX_TTS_CN_URL]
    effective_endpoint = voice.get_minimax_tts_endpoint()
    if effective_endpoint not in minimax_tts_endpoints:
        effective_endpoint = voice.MINIMAX_TTS_GLOBAL_URL
    minimax_tts_base_url = stable_selectbox(
        tr("MiniMax TTS Endpoint"),
        options=minimax_tts_endpoints,
        default_value=effective_endpoint,
        key="minimax_tts_endpoint_select",
        # Reusing LLM Key requires tracking LLM region to prevent the UI offering options that will not take effect;
        # entering an independent TTS Key allows selecting sites individually.
        disabled=not dedicated_key,
    )
    if dedicated_key:
        _set_runtime_config("minimax_tts", "base_url", minimax_tts_base_url)

    configured_model = config.minimax_tts.get(
        "model_id", voice.MINIMAX_TTS_DEFAULT_MODEL
    )
    if configured_model not in voice.MINIMAX_TTS_MODELS:
        configured_model = voice.MINIMAX_TTS_DEFAULT_MODEL
    minimax_tts_model = stable_selectbox(
        tr("MiniMax TTS Model"),
        options=list(voice.MINIMAX_TTS_MODELS),
        default_value=configured_model,
        key="minimax_tts_model_select",
    )
    _set_runtime_config("minimax_tts", "model_id", minimax_tts_model)

    if st.button(
        tr("Load MiniMax Voices"),
        key="load_minimax_voices_button",
        icon=":material/refresh:",
        use_container_width=True,
    ):
        try:
            available_voices = voice.get_minimax_voice_catalog(
                api_key=effective_api_key,
                endpoint=minimax_tts_base_url,
                voice_type="all",
            )
        except Exception as exc:
            # Expose exceptions to users and write logs. Account region mismatches, insufficient key permissions,
            # or network failures are common; silently returning an empty list misleads users into thinking no voices exist.
            logger.warning(f"load MiniMax voices failed: {exc}")
            st.error(tr("MiniMax Voices Load Failed").format(error=str(exc)))
        else:
            _cache_minimax_voices(
                effective_api_key,
                minimax_tts_base_url,
                available_voices,
            )
            st.success(tr("MiniMax Voices Loaded").format(count=len(available_voices)))

    available_voices = _get_cached_minimax_voices(
        effective_api_key,
        minimax_tts_base_url,
    )
    voice_labels = {
        f"minimax:{item['voice_id']}": (
            f"{item['voice_name']} ({item['voice_id']})"
            if item["voice_name"] != item["voice_id"]
            else item["voice_id"]
        )
        for item in available_voices
    }
    configured_voice_id = str(
        config.minimax_tts.get("voice_id", voice.MINIMAX_TTS_DEFAULT_VOICE)
        or voice.MINIMAX_TTS_DEFAULT_VOICE
    ).strip()
    configured_voice = f"minimax:{configured_voice_id}"
    # Retain current Voice ID when voices haven't been fetched, interface is temporarily unavailable, or config uses
    # custom cloned voices outside the list, ensuring generation does not strictly depend on remote query results.
    voice_labels.setdefault(configured_voice, configured_voice_id)
    return list(voice_labels), voice_labels


def _sync_elevenlabs_api_key_input():
    """
    Synchronize ElevenLabs password widget, persisted config, and environment variables, returning valid key.

    When a browser tab reconnects to a restarted server, Streamlit may replay an empty password widget state.
    Because this empty state cannot be reliably distinguished from user clearing, prioritize restoring valid keys
    from config or environment variables, preventing empty state overwriting and ensuring immediate voice loading.
    To completely remove a key, modify config files or environment variables to avoid reconnection misclassification.
    """
    widget_key = "elevenlabs_api_key_input"
    configured_key = str(config.elevenlabs.get("api_key", "") or "").strip()
    env_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    effective_key = configured_key or env_key
    had_widget_state = widget_key in st.session_state
    entered_key = str(st.session_state.get(widget_key, "") or "").strip()

    if not entered_key and effective_key:
        # Reconnection empty states must not overwrite valid credentials and must be restored before rendering voice lists;
        # otherwise, current page requests ElevenLabs with an empty key even though config was not cleared.
        st.session_state[widget_key] = effective_key
        entered_key = effective_key
        if had_widget_state:
            logger.debug("restored ElevenLabs API key after empty session replay")
    elif not had_widget_state:
        # Initialize before widget creation to avoid Streamlit default value conflict warnings when passing both value
        # and session_state; initialize to empty when no keys exist.
        st.session_state[widget_key] = entered_key

    if entered_key and entered_key != effective_key:
        # Only newly entered user values are written to config.toml. Environment variables used as valid values are not copied
        # to disk, keeping keys injected by containers or deployment platforms isolated to the runtime environment.
        for cache_key in list(st.session_state.keys()):
            if str(cache_key).startswith("elevenlabs_voices_"):
                del st.session_state[cache_key]
        _set_runtime_config("elevenlabs", "api_key", entered_key)

    return entered_key


def _sync_voxcpm_api_key_input():
    """Restore VoxCPM password widget from empty states replayed by Streamlit upon reconnection."""
    widget_key = "voxcpm_api_key_input"
    configured_key = str(config.voxcpm.get("api_key", "") or "").strip()
    had_widget_state = widget_key in st.session_state
    entered_key = str(st.session_state.get(widget_key, "") or "").strip()

    if not entered_key and configured_key:
        # Browser reconnection may replay empty password states. Retain saved credentials to prevent this rerun
        # from overwriting valid keys in config.toml with empty strings via _set_runtime_config.
        st.session_state[widget_key] = configured_key
        entered_key = configured_key
        if had_widget_state:
            logger.debug("restored VoxCPM API key after empty session replay")
    elif not had_widget_state:
        st.session_state[widget_key] = entered_key

    return entered_key


def _render_elevenlabs_api_key_input(label_key):
    """
    Render unified API key input shared between ElevenLabs TTS and BGM.

    Using two separate widget keys for TTS and BGM on the same page causes Streamlit to retain old values separately,
    allowing later inputs to overwrite shared config. Use a single key and centralize environment variable backfill,
    config updates, and voice cache invalidation to ensure UI display and background tasks always read the same value.
    """
    _sync_elevenlabs_api_key_input()
    return st.text_input(
        tr(label_key),
        type="password",
        key="elevenlabs_api_key_input",
    ).strip()


def _render_background_music_settings(params, elevenlabs_api_key_rendered=False):
    """Render background music source and volume settings, returning uploaded files to be saved."""
    uploaded_bgm_file = None
    previous_bgm_type = st.session_state.get("last_rendered_bgm_type")
    st.divider()
    bgm_options = [
        (tr("No Background Music"), ""),
        (tr("Random Background Music"), "random"),
        (tr("Preset Song"), "preset"),
        (tr("Custom Background Music"), "custom"),
        (tr("Sonilo Background Music"), "sonilo"),
        (tr("ElevenLabs Background Music"), "elevenlabs"),
    ]
    selected_bgm_type = stable_selectbox(
        tr("Background Music Source"),
        options=[value for _, value in bgm_options],
        default_value=_saved_ui_choice(
            "bgm_type",
            [value for _, value in bgm_options],
            "random",
        ),
        key="bgm_type_select",
        format_func=lambda value: dict((v, label) for label, v in bgm_options)[value],
    )
    params.bgm_type = selected_bgm_type
    _set_runtime_config("ui", "bgm_type", params.bgm_type)
    if params.bgm_type == "sonilo":
        configured_key = str(config.app.get("sonilo_api_key", "") or "").strip()
        effective_key = configured_key or os.getenv("SONILO_API_KEY", "").strip()
        entered_key = st.text_input(
            tr("Sonilo API Key"),
            value=effective_key,
            type="password",
            key="sonilo_api_key_input",
        ).strip()
        # Users require configured keys backfilled into password inputs. Configuration values take precedence over env vars;
        # write back only when users modify inputs or originally use config, avoiding copying env keys into config.toml on no-op.
        if configured_key or entered_key != effective_key:
            _set_runtime_config("app", "sonilo_api_key", entered_key)
    elif params.bgm_type == "elevenlabs":
        if elevenlabs_api_key_rendered:
            # Avoid creating a second widget if the TTS area already rendered the shared input, preventing conflicting session states.
            # Explanatory text guides users to the shared configuration above.
            st.caption(tr("ElevenLabs API Key Help"))
        else:
            _render_elevenlabs_api_key_input("ElevenLabs Music API Key")

    bgm_volume_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    params.bgm_volume = stable_selectbox(
        tr("Background Music Volume"),
        options=bgm_volume_options,
        default_value=_saved_ui_choice("bgm_volume", bgm_volume_options, 0.2),
        key="bgm_volume_select",
        format_func=lambda value: f"{int(value * 100)}%",
        disabled=not params.bgm_type,
    )
    _set_runtime_config("ui", "bgm_volume", params.bgm_volume)
    bgm_enabled = bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)

    if params.bgm_type == "custom":
        uploaded_bgm_file = st.file_uploader(
            tr("Upload Background Music"),
            type=[
                extension.removeprefix(".")
                for extension in bgm_service.SUPPORTED_BGM_EXTENSIONS
            ],
            accept_multiple_files=False,
            key="custom_bgm_uploader",
            help=tr("Upload Background Music Help"),
            # Streamlit displays a global 200MB limit on widgets by default. Keep aligned with the service layer's
            # 30MB hard limit to avoid accepting selections in the UI only to reject them on server submission.
            max_upload_size=bgm_service.MAX_BGM_UPLOAD_BYTES // (1024 * 1024),
        )
        if uploaded_bgm_file is not None and bgm_enabled:
            try:
                safe_name = bgm_service.sanitize_upload_filename(uploaded_bgm_file.name)
                # Streamlit reruns the page on adjusting volume or other controls. Use content hashes to distinguish uploads
                # and cache full decoding results in the session, preventing stale results on same-name/same-size files
                # while avoiding repeated FFmpeg invocations on every rerun.
                validation_key = (
                    safe_name,
                    uploaded_bgm_file.size,
                    hashlib.sha256(uploaded_bgm_file.getbuffer()).hexdigest(),
                )
                cached_validation = st.session_state.get("custom_bgm_validation")
                if (
                    not cached_validation
                    or cached_validation.get("key") != validation_key
                ):
                    try:
                        bgm_service.validate_bgm_upload(
                            uploaded_bgm_file.name, uploaded_bgm_file
                        )
                    except bgm_service.BgmUploadError as exc:
                        cached_validation = {
                            "key": validation_key,
                            "error": str(exc),
                            "error_type": "upload",
                        }
                        # Failed validation results for the same fingerprint enter session cache; log only once
                        # on initial validation run to avoid cluttering logs during regular widget reruns.
                        logger.warning(
                            "WebUI background music validation rejected: "
                            f"name={safe_name}, error={str(exc)}"
                        )
                    except bgm_service.BgmServiceError as exc:
                        cached_validation = {
                            "key": validation_key,
                            "error": str(exc),
                            "error_type": "service",
                        }
                        logger.error(
                            "WebUI background music validation failed: "
                            f"name={safe_name}, error={str(exc)}"
                        )
                    else:
                        cached_validation = {
                            "key": validation_key,
                            "error": "",
                            "error_type": "",
                        }
                    st.session_state["custom_bgm_validation"] = cached_validation

                if cached_validation.get("error"):
                    if cached_validation.get("error_type") == "service":
                        raise bgm_service.BgmServiceError(cached_validation["error"])
                    raise bgm_service.BgmUploadError(cached_validation["error"])
            except bgm_service.BgmUploadError:
                # Invalid files cannot reuse the previous valid upload filename, which could leave task parameters pointing to stale BGM.
                # Retain UploadedFile return value so clicking Generate triggers final server-side validation rather than silently omitting BGM.
                params.bgm_file = ""
                st.error(tr("Invalid Background Music"))
            except bgm_service.BgmServiceError:
                params.bgm_file = ""
                st.error(tr("Background Music Validation Failed"))
            else:
                # Display player and 'Ready' badge only after full decode validation passes. Files are persisted only upon clicking Generate;
                # previewing or removing files does not pollute storage/bgm.
                uploaded_mime_type = str(getattr(uploaded_bgm_file, "type", "") or "")
                preview_mime_type = (
                    uploaded_mime_type
                    if uploaded_mime_type.startswith("audio/")
                    else mimetypes.guess_type(safe_name)[0] or "audio/mpeg"
                )
                st.audio(uploaded_bgm_file, format=preview_mime_type)
                st.info(f"{tr('Background Music Ready')}: {safe_name}")
                params.bgm_file = safe_name

        # Streamlit cleans up widget state when conditional controls are temporarily omitted from rendering.
        # Restore persisted values when switching back from other BGM sources; user clearing within the same source
        # keeps previous_bgm_type unchanged, avoiding rebounding to old values.
        if previous_bgm_type != "custom":
            st.session_state["custom_bgm_file_input"] = _saved_ui_text(
                "custom_bgm_file"
            )
        custom_bgm_file = st.text_input(
            tr("Custom Background Music File"),
            key="custom_bgm_file_input",
            disabled=uploaded_bgm_file is not None,
        )
        _set_runtime_config(
            "ui", "custom_bgm_file", custom_bgm_file.strip()
        )
        if uploaded_bgm_file is None and custom_bgm_file and bgm_enabled:
            # Filenames are validated by the service layer after mapping to storage/bgm or resource/songs;
            # UI rejects arbitrary paths outside the two whitelisted directories.
            params.bgm_file = custom_bgm_file.strip()
        elif not bgm_enabled:
            # Upload widget retains user-selected files; raising volume automatically triggers full validation on the next rerun.
            # Clear current task parameters to prevent 0-volume tasks from saving or parsing this file.
            params.bgm_file = ""

    if params.bgm_type == "preset":
        # The service layer standardizes extensions, temporary files, and symlink validation.
        # Reuse its results directly, preventing the UI from maintaining duplicate enumeration rules or formatting divergences.
        available_song_paths = bgm_service.list_builtin_bgm_files()
        songs_by_name = {
            os.path.basename(song_path): song_path for song_path in available_song_paths
        }
        available_songs = list(songs_by_name)
        if not available_songs:
            st.warning(tr("No Background Music Available"))
            params.bgm_file = ""
        else:
            default_preset_song = _saved_ui_text("preset_song", available_songs[0])
            requested_preset_song = st.session_state.get(
                localized_widget_key("preset_song_select"), default_preset_song
            )
            if requested_preset_song not in available_songs:
                # Historical tasks or settings exported from other versions may reference songs absent from the current installation.
                # Display explicit notice and fall back to the first song via stable_selectbox to avoid silent substitution.
                st.warning(tr("Selected Background Music Unavailable"))
            selected_song = stable_selectbox(
                tr("Preset Song"),
                options=available_songs,
                default_value=(
                    default_preset_song
                    if default_preset_song in available_songs
                    else available_songs[0]
                ),
                key="preset_song_select",
            )
            _set_runtime_config("ui", "preset_song", selected_song)
            # Provide online preview immediately upon song selection. The player reads verified paths
            # produced by service layer whitelist validation, rejecting arbitrary page-supplied paths.
            selected_song_path = songs_by_name[selected_song]
            preview_mime_type = (
                mimetypes.guess_type(selected_song_path)[0] or "audio/mpeg"
            )
            preview_available = True
            try:
                # Streamlit wraps OSError into internal exceptions on failed path reads, preventing file error handling below.
                # Read bytes manually first to preserve player behavior and handle transient Docker mount drops or permission shifts stably.
                selected_song_bytes = Path(selected_song_path).read_bytes()
            except OSError as exc:
                preview_available = False
                # Files may be deleted by other processes after enumeration. Preview failures must not break page rendering
                # or video parameter editing, but require logging to diagnose environment and mount issues.
                logger.warning(
                    "failed to preview preset background music: "
                    f"name={selected_song}, error={str(exc)}"
                )
                st.warning(tr("Background Music Preview Failed"))
            else:
                st.audio(selected_song_bytes, format=preview_mime_type)
            if bgm_enabled and preview_available:
                params.bgm_file = selected_song
            else:
                params.bgm_file = ""

    if params.bgm_type == "sonilo":
        if previous_bgm_type != "sonilo":
            st.session_state["sonilo_bgm_prompt_input"] = _saved_ui_text(
                "sonilo_bgm_prompt",
                max_length=sonilo_service.MAX_PROMPT_LENGTH,
            )
        params.video_music_prompt = st.text_input(
            tr("Sonilo Music Prompt"),
            key="sonilo_bgm_prompt_input",
            max_chars=sonilo_service.MAX_PROMPT_LENGTH,
            help=tr("Sonilo Music Prompt Help"),
        ).strip()
        _set_runtime_config(
            "ui", "sonilo_bgm_prompt", params.video_music_prompt
        )
        if params.video_count > 1:
            st.warning(tr("Sonilo Multiple Videos Warning"))
        if st.button(
            tr("Test Sonilo Connection"),
            key="test_sonilo_connection_button",
            use_container_width=True,
        ):
            try:
                sonilo_service.test_connection()
            except sonilo_service.SoniloError as exc:
                logger.warning(f"Sonilo connection test failed: {exc}")
                st.error(tr("Sonilo Connection Test Failed").format(error=str(exc)))
            else:
                st.success(tr("Sonilo Connection Test Succeeded"))
    elif params.bgm_type == "elevenlabs":
        if previous_bgm_type != "elevenlabs":
            st.session_state["elevenlabs_music_prompt_input"] = _saved_ui_text(
                "elevenlabs_music_prompt",
                max_length=elevenlabs_music_service.MAX_PROMPT_LENGTH,
            )
        params.video_music_prompt = st.text_input(
            tr("ElevenLabs Music Prompt"),
            key="elevenlabs_music_prompt_input",
            max_chars=elevenlabs_music_service.MAX_PROMPT_LENGTH,
            help=tr("ElevenLabs Music Prompt Help"),
        ).strip()
        _set_runtime_config(
            "ui", "elevenlabs_music_prompt", params.video_music_prompt
        )
        if params.video_count > 1:
            st.warning(tr("ElevenLabs Multiple Videos Warning"))
        if st.button(
            tr("Test ElevenLabs Connection"),
            key="test_elevenlabs_music_connection_button",
            use_container_width=True,
        ):
            try:
                elevenlabs_music_service.test_connection()
            except elevenlabs_music_service.ElevenLabsPaidPlanRequiredError:
                st.error(tr("ElevenLabs Paid Plan Required"))
            except elevenlabs_music_service.ElevenLabsMusicError as exc:
                logger.warning(f"ElevenLabs connection test failed: {exc}")
                st.error(tr("ElevenLabs Connection Test Failed").format(error=str(exc)))
            else:
                st.success(tr("ElevenLabs Connection Test Succeeded"))
    if params.bgm_type == "sonilo" and bgm_enabled and not sonilo_service.is_enabled():
        # Task layer skips generating or mixing Sonilo music when volume is 0, eliminating key requirements;
        # shares service layer rules with task entry points to avoid divergent UI hints and runtime conditions.
        st.warning(tr("Sonilo API Key Required"))
    elif (
        params.bgm_type == "elevenlabs"
        and bgm_enabled
        and not elevenlabs_music_service.is_enabled()
    ):
        st.warning(tr("ElevenLabs API Key Required"))
    st.session_state["last_rendered_bgm_type"] = params.bgm_type
    return uploaded_bgm_file


def _render_audio_settings(panel, params):
    """Render audio settings, returning uploaded audio and current voiceover mode."""
    with panel:
        with st.container(border=True):
            st.write(tr("Audio Settings"))

            # Voice mode is top-level state in audio settings, distinguishing auto TTS, user upload, and no voiceover.
            # Legacy configs lacking voice_mode maintain compatibility via original tts_server no-voice sentinels.
            saved_tts_server = config.ui.get("tts_server", "azure-tts-v1")
            saved_voice_mode = config.ui.get("voice_mode")
            if saved_voice_mode not in {
                VOICE_MODE_TTS,
                VOICE_MODE_UPLOAD,
                VOICE_MODE_NONE,
            }:
                saved_voice_mode = (
                    VOICE_MODE_NONE
                    if saved_tts_server == voice.NO_VOICE_NAME
                    else VOICE_MODE_TTS
                )
            voice_mode_options = [VOICE_MODE_TTS, VOICE_MODE_UPLOAD, VOICE_MODE_NONE]
            voice_mode_labels = {
                VOICE_MODE_TTS: tr("Automatic Voiceover"),
                VOICE_MODE_UPLOAD: tr("Upload Voiceover"),
                VOICE_MODE_NONE: tr("No Voiceover"),
            }
            voice_mode = stable_segmented_control(
                tr("Voiceover Mode"),
                options=voice_mode_options,
                default_value=saved_voice_mode,
                key="voice_mode_control",
                format_func=lambda value: voice_mode_labels[value],
                width="stretch",
            )
            _set_runtime_config("ui", "voice_mode", voice_mode)
            tts_mode_enabled = voice_mode == VOICE_MODE_TTS

            # Provider dropdown selects auto TTS services only; no voiceover is handled by the mode above
            # and is no longer mixed into TTS providers, avoiding duplicate controls for the same state.
            tts_servers = [
                ("azure-tts-v1", "Azure TTS V1 (Edge TTS)"),
                ("azure-tts-v2", "Azure TTS V2"),
                ("siliconflow", "SiliconFlow TTS"),
                ("gemini-tts", "Google Gemini TTS"),
                ("mimo-tts", "Xiaomi MiMo TTS"),
                ("minimax-tts", "MiniMax TTS"),
                ("elevenlabs", "ElevenLabs TTS"),
                ("chatterbox", "Chatterbox TTS"),
                ("kokoro", "Kokoro TTS"),
                ("fish_audio", "Fish Audio TTS"),
                ("voxcpm", "VoxCPM TTS"),
            ]

            tts_server_values = [server_value for server_value, _ in tts_servers]
            if saved_tts_server not in tts_server_values:
                saved_tts_server = "azure-tts-v1"

            if tts_mode_enabled:
                selected_tts_server = stable_selectbox(
                    tr("Voiceover Service"),
                    options=tts_server_values,
                    default_value=saved_tts_server,
                    key="tts_server_select",
                    format_func=lambda value: dict(
                        (v, label) for v, label in tts_servers
                    )[value],
                )
            else:
                # Non-auto voice modes do not render TTS widgets, but retain previous selections to resume when switched back.
                selected_tts_server = saved_tts_server

            _set_runtime_config("ui", "tts_server", selected_tts_server)

            # Service instructions follow provider selection, explaining prerequisites before voice and credential configuration.
            # Providers without instructions do not render empty callout blocks.
            if tts_mode_enabled:
                provider_tips = get_tts_provider_tips(selected_tts_server)
                if provider_tips:
                    st.info(provider_tips)

            # MiniMax reuses the generic 'Voiceover Voice' selector below. Provider config function refreshes remote voices
            # and returns friendly names without rendering redundant Voice ID or voice dropdown widgets.
            minimax_voices = []
            minimax_voice_labels = {}
            if tts_mode_enabled and selected_tts_server == "minimax-tts":
                minimax_voices, minimax_voice_labels = _render_minimax_tts_settings()

            # Fetch voice list based on selected TTS server
            filtered_voices = []
            saved_voice_name = config.ui.get("voice_name", "")
            elevenlabs_api_key_rendered = False

            if not tts_mode_enabled:
                # Uploaded audio and no voiceover modes skip loading remote voices, reducing needless network calls and UI noise.
                filtered_voices = []
            elif selected_tts_server == "siliconflow":
                # Get SiliconFlow voice list
                filtered_voices = voice.get_siliconflow_voices()
            elif selected_tts_server == "gemini-tts":
                # Get Gemini TTS voice list
                filtered_voices = voice.get_gemini_voices()
            elif selected_tts_server == "mimo-tts":
                # Get preset voice list for Xiaomi MiMo TTS
                filtered_voices = voice.get_mimo_voices()
            elif selected_tts_server == "minimax-tts":
                filtered_voices = minimax_voices
            elif selected_tts_server == "elevenlabs":
                # Voice list renders before key input; restore reconnection state and read config/env first,
                # otherwise the page loads and caches an empty voice list using an empty key.
                saved_elevenlabs_api_key = _sync_elevenlabs_api_key_input()
                cache_key = f"elevenlabs_voices_{saved_elevenlabs_api_key}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = voice.get_elevenlabs_voices(
                        saved_elevenlabs_api_key
                    )
                filtered_voices = st.session_state[cache_key]
            elif selected_tts_server == "chatterbox":
                # Preset voices for self-hosted Chatterbox service (from [chatterbox] voices config)
                _sync_chatterbox_config_from_session_state()
                filtered_voices = voice.get_chatterbox_voices()
            elif selected_tts_server == "kokoro":
                # Self-hosted Kokoro service voices: fetch from server /audio/voices when [kokoro] voices is empty
                _sync_kokoro_config_from_session_state()
                filtered_voices = _get_kokoro_voice_options(saved_voice_name)
            elif selected_tts_server == "fish_audio":
                filtered_voices = voice.get_fish_audio_voices()
            elif selected_tts_server == "voxcpm":
                filtered_voices = voice.get_voxcpm_voices()
            else:
                # Get Azure voice list
                all_voices = voice.get_all_azure_voices(filter_locals=None)

                # Filter voices based on selected TTS server
                for v in all_voices:
                    if selected_tts_server == "azure-tts-v2":
                        # V2 version voice names contain 'v2'
                        if "V2" in v:
                            filtered_voices.append(v)
                    else:
                        # V1 version voice names do not contain 'v2'
                        if "V2" not in v:
                            filtered_voices.append(v)

            def _friendly(v):
                if voice.is_no_voice(v):
                    return tr("No Voice Selected")
                if voice.is_elevenlabs_voice(v):
                    parts = v.split(":", 2)
                    return parts[2] if len(parts) >= 3 else v
                if voice.is_chatterbox_voice(v) or voice.is_kokoro_voice(v):
                    name = v.split(":", 1)[1] if ":" in v else v
                    return name.replace("-Female", "").replace("-Male", "")
                if voice.is_minimax_voice(v):
                    return minimax_voice_labels.get(v, v.split(":", 1)[1])
                if voice.is_fish_audio_voice(v):
                    parts = v.split(":", 2)
                    display_name = parts[2] if len(parts) >= 3 else v
                    return (
                        display_name.replace("Female", tr("Female"))
                        .replace("Male", tr("Male"))
                    )
                if voice.is_voxcpm_voice(v):
                    return v.split(":", 1)[1] or DEFAULT_VOXCPM_VOICE
                return (
                    v.replace("Female", tr("Female"))
                    .replace("Male", tr("Male"))
                    .replace("Neural", "")
                )

            friendly_names = {v: _friendly(v) for v in filtered_voices}

            # Gemini legacy catalogs placed inferred gender in values (e.g. Charon-Male).
            # Map to new official style values by base voice name, preserving previous selections upon upgrading.
            if (
                selected_tts_server == "gemini-tts"
                and saved_voice_name not in friendly_names
            ):
                saved_gemini_voice = voice.parse_gemini_voice_name(saved_voice_name)
                saved_voice_name = next(
                    (
                        candidate
                        for candidate in filtered_voices
                        if voice.parse_gemini_voice_name(candidate)
                        == saved_gemini_voice
                    ),
                    saved_voice_name,
                )

            saved_voice_name_index = 0

            # Check if saved voice is in currently filtered voice list
            if saved_voice_name in friendly_names:
                saved_voice_name_index = list(friendly_names.keys()).index(
                    saved_voice_name
                )
            else:
                # If not, select a default voice based on current UI language
                for i, v in enumerate(filtered_voices):
                    if v.lower().startswith(st.session_state["ui_language"].lower()):
                        saved_voice_name_index = i
                        break

            # Fall back to first voice if no matching voice is found
            if saved_voice_name_index >= len(friendly_names) and friendly_names:
                saved_voice_name_index = 0

            # Ensure voices are available
            if tts_mode_enabled and friendly_names:
                voice_name = stable_selectbox(
                    tr("Voiceover Voice"),
                    options=list(friendly_names.keys()),
                    default_value=list(friendly_names.keys())[saved_voice_name_index],
                    key=f"speech_synthesis_select_{selected_tts_server}",
                    format_func=lambda value: friendly_names.get(
                        value,
                        str(value).removeprefix("minimax:"),
                    ),
                    # MiniMax supports direct manual entry of custom/cloned Voice IDs outside the list;
                    # other providers keep standard selector behavior to limit scope of change.
                    accept_new_options=selected_tts_server == "minimax-tts",
                )

                if selected_tts_server == "minimax-tts":
                    custom_voice_id = str(voice_name or "").strip()
                    if custom_voice_id and not voice.is_minimax_voice(custom_voice_id):
                        voice_name = f"minimax:{custom_voice_id}"
                    if voice.is_minimax_voice(voice_name):
                        _set_runtime_config(
                            "minimax_tts",
                            "voice_id",
                            voice_name.split(":", 1)[1],
                        )

                params.voice_name = voice_name
                if not voice.is_no_voice(voice_name):
                    # Placeholder sentinel is used solely for disabled display in non-auto modes without overriding
                    # user's real selection, allowing settings to resume when switching back to auto voiceover.
                    _set_runtime_config("ui", "voice_name", voice_name)
            elif tts_mode_enabled:
                # Display notice if no voices are available
                st.warning(
                    tr(
                        "No voices available for the selected TTS server. Please select another server."
                    )
                )
                voice_name = ""
                params.voice_name = ""
                _set_runtime_config("ui", "voice_name", "")
            else:
                # Non-auto voiceover modes hide voice widgets, reusing stored values to maintain stable parameter structure.
                voice_name = saved_voice_name or voice.NO_VOICE_NAME
                params.voice_name = voice_name

            # Display service region and API key input when V2 or V2 voice is selected
            if tts_mode_enabled and (
                selected_tts_server == "azure-tts-v2"
                or (voice_name and voice.is_azure_v2_voice(voice_name))
            ):
                saved_azure_speech_region = config.azure.get("speech_region", "")
                saved_azure_speech_key = config.azure.get("speech_key", "")
                azure_speech_region = st.text_input(
                    tr("Speech Region"),
                    value=saved_azure_speech_region,
                    key="azure_speech_region_input",
                )
                azure_speech_key = st.text_input(
                    tr("Speech Key"),
                    value=saved_azure_speech_key,
                    type="password",
                    key="azure_speech_key_input",
                )
                _set_runtime_config("azure", "speech_region", azure_speech_region)
                _set_runtime_config("azure", "speech_key", azure_speech_key)

            if tts_mode_enabled and selected_tts_server == "gemini-tts":
                # Gemini TTS shares the same key with Gemini LLM; providing direct entry in the audio panel
                # allows users to configure voice without having to switch LLM provider first.
                gemini_tts_api_key = st.text_input(
                    tr("Gemini API Key"),
                    value=config.app.get("gemini_api_key", ""),
                    type="password",
                    key="gemini_tts_api_key_input",
                )
                _set_runtime_config("app", "gemini_api_key", gemini_tts_api_key)

            # Display API key input and instructions when SiliconFlow is selected
            if tts_mode_enabled and (
                selected_tts_server == "siliconflow"
                or (voice_name and voice.is_siliconflow_voice(voice_name))
            ):
                saved_siliconflow_api_key = config.siliconflow.get("api_key", "")

                siliconflow_api_key = st.text_input(
                    tr("SiliconFlow API Key"),
                    value=saved_siliconflow_api_key,
                    type="password",
                    key="siliconflow_api_key_input",
                )

                _set_runtime_config("siliconflow", "api_key", siliconflow_api_key)

            # When Xiaomi MiMo TTS is selected, reuse MiMo LLM provider's API key.
            # Users generating both script copy and speech with MiMo only need to maintain one key.
            if tts_mode_enabled and (
                selected_tts_server == "mimo-tts"
                or (voice_name and voice.is_mimo_voice(voice_name))
            ):
                saved_mimo_api_key = config.app.get("mimo_api_key", "")

                mimo_api_key = st.text_input(
                    tr("MiMo API Key"),
                    value=saved_mimo_api_key,
                    type="password",
                    key="mimo_tts_api_key_input",
                )

                _set_runtime_config("app", "mimo_api_key", mimo_api_key)

            # ElevenLabs API key section
            if tts_mode_enabled and (
                selected_tts_server == "elevenlabs"
                or (voice_name and voice.is_elevenlabs_voice(voice_name))
            ):
                _render_elevenlabs_api_key_input(
                    "ElevenLabs API Key",
                )
                elevenlabs_api_key_rendered = True

                _elevenlabs_models = [
                    "eleven_multilingual_v2",
                    "eleven_flash_v2_5",
                    "eleven_v3",
                ]
                saved_elevenlabs_model = config.elevenlabs.get(
                    "model_id", "eleven_multilingual_v2"
                )
                if saved_elevenlabs_model not in _elevenlabs_models:
                    saved_elevenlabs_model = "eleven_multilingual_v2"
                elevenlabs_model = stable_selectbox(
                    tr("ElevenLabs Model"),
                    options=_elevenlabs_models,
                    default_value=saved_elevenlabs_model,
                    key="elevenlabs_model_select",
                )
                _set_runtime_config("elevenlabs", "model_id", elevenlabs_model)

            # Fish Audio API settings section
            if tts_mode_enabled and (
                selected_tts_server == "fish_audio"
                or (voice_name and voice.is_fish_audio_voice(voice_name))
            ):
                saved_fish_api_key = (
                    config.fish_audio.get("api_key", "")
                    if hasattr(config, "fish_audio") and isinstance(config.fish_audio, dict)
                    else ""
                )
                fish_audio_api_key = st.text_input(
                    tr("Fish Audio API Key"),
                    value=saved_fish_api_key,
                    type="password",
                    key="fish_audio_api_key_input",
                )
                _set_runtime_config("fish_audio", "api_key", fish_audio_api_key)

                _fish_audio_models = [
                    "s2.1-pro-free",
                    "s2.1-pro",
                    "s2-pro",
                ]
                saved_fish_model = (
                    config.fish_audio.get("model", "s2.1-pro-free")
                    if hasattr(config, "fish_audio") and isinstance(config.fish_audio, dict)
                    else "s2.1-pro-free"
                )
                if saved_fish_model not in _fish_audio_models:
                    saved_fish_model = "s2.1-pro-free"
                fish_model = stable_selectbox(
                    tr("Fish Audio Model"),
                    options=_fish_audio_models,
                    default_value=saved_fish_model,
                    key="fish_audio_model_select",
                )
                _set_runtime_config("fish_audio", "model", fish_model)

            # ModelBest hosts VoxCPM behind its streaming Audio Speech API.
            # The fixed provider endpoint is still editable for compatible
            # gateways, while the user only has to supply an API key and a
            # speech_synthesis-capable model id for the standard platform.
            if tts_mode_enabled and (
                selected_tts_server == "voxcpm"
                or (voice_name and voice.is_voxcpm_voice(voice_name))
            ):
                _sync_voxcpm_api_key_input()
                voxcpm_api_key = st.text_input(
                    tr("VoxCPM API Key"),
                    type="password",
                    key="voxcpm_api_key_input",
                )
                _set_runtime_config("voxcpm", "api_key", voxcpm_api_key)

                voxcpm_model = st.text_input(
                    tr("VoxCPM Model ID"),
                    value=config.voxcpm.get("model_id", ""),
                    key="voxcpm_model_id_input",
                    placeholder=tr("VoxCPM Model ID Placeholder"),
                )
                _set_runtime_config("voxcpm", "model_id", voxcpm_model.strip())

                voxcpm_base_url = st.text_input(
                    tr("VoxCPM Base URL"),
                    value=config.voxcpm.get("base_url") or DEFAULT_VOXCPM_BASE_URL,
                    key="voxcpm_base_url_input",
                    placeholder=DEFAULT_VOXCPM_BASE_URL,
                )
                _set_runtime_config(
                    "voxcpm",
                    "base_url",
                    (voxcpm_base_url or DEFAULT_VOXCPM_BASE_URL).strip().rstrip("/"),
                )

                uploaded_reference_audio = st.file_uploader(
                    tr("VoxCPM Reference Audio"),
                    type=list(voice.VOXCPM_REFERENCE_AUDIO_FILE_TYPES),
                    key="voxcpm_reference_audio_uploader",
                    help=tr("VoxCPM Reference Audio Help"),
                )
                reference_audio = _sync_voxcpm_reference_audio(
                    uploaded_reference_audio
                )
                st.caption(tr("VoxCPM Reference Audio Notice"))
                reference_audio_error = st.session_state.get(
                    VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY
                )
                if reference_audio_error:
                    st.error(
                        tr("VoxCPM Reference Audio Invalid").format(
                            error=reference_audio_error
                        )
                    )

                if reference_audio:
                    high_fidelity_enabled = st.toggle(
                        tr("VoxCPM High Fidelity Delivery"),
                        key=VOXCPM_HIGH_FIDELITY_SESSION_KEY,
                        help=tr("VoxCPM High Fidelity Delivery Help"),
                    )
                    if high_fidelity_enabled:
                        use_separate_prompt_audio = st.toggle(
                            tr("VoxCPM Separate Prompt Audio"),
                            key=VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY,
                            help=tr("VoxCPM Separate Prompt Audio Help"),
                        )
                        _sync_voxcpm_prompt_example_mode(use_separate_prompt_audio)
                        if use_separate_prompt_audio:
                            prompt_audio_file = st.file_uploader(
                                tr("VoxCPM Prompt Audio"),
                                type=list(voice.VOXCPM_REFERENCE_AUDIO_FILE_TYPES),
                                key="voxcpm_prompt_audio_uploader",
                                help=tr("VoxCPM Prompt Audio Help"),
                            )
                            _sync_voxcpm_prompt_audio(prompt_audio_file)
                        else:
                            _clear_voxcpm_separate_prompt_audio()
                        effective_prompt_audio = _get_voxcpm_effective_prompt_audio()
                        if st.button(
                            tr("Transcribe VoxCPM Prompt Audio"),
                            key="transcribe_voxcpm_prompt_audio_button",
                            icon=":material/transcribe:",
                            help=tr("Transcribe VoxCPM Prompt Audio Help"),
                            use_container_width=True,
                            disabled=not bool(effective_prompt_audio),
                        ):
                            try:
                                with st.spinner(tr("Transcribing VoxCPM Prompt Audio")):
                                    recognized_text = subtitle.transcribe_audio_bytes(
                                        effective_prompt_audio
                                    )
                            except Exception:
                                logger.exception(
                                    "failed to transcribe VoxCPM prompt audio"
                                )
                                recognized_text = ""
                            if recognized_text:
                                st.session_state[VOXCPM_PROMPT_TEXT_SESSION_KEY] = (
                                    recognized_text
                                )
                                st.toast(tr("VoxCPM Prompt Audio Transcribed"))
                            else:
                                st.error(tr("VoxCPM Prompt Audio Transcription Failed"))
                        st.text_area(
                            tr("VoxCPM Prompt Text"),
                            key=VOXCPM_PROMPT_TEXT_SESSION_KEY,
                            help=tr("VoxCPM Prompt Text Help"),
                            height=100,
                        )
                        st.caption(tr("VoxCPM Prompt Transcript Review"))
                        prompt_error = _get_voxcpm_prompt_validation_error()
                        if prompt_error:
                            st.error(
                                tr("VoxCPM Prompt Invalid").format(
                                    error=prompt_error
                                )
                            )
                    else:
                        _clear_voxcpm_prompt_state()

            # Chatterbox API settings section (self-hosted, OpenAI-compatible)
            if tts_mode_enabled and (
                selected_tts_server == "chatterbox"
                or (voice_name and voice.is_chatterbox_voice(voice_name))
            ):
                chatterbox_base_url = st.text_input(
                    tr("Chatterbox Base URL"),
                    value=config.chatterbox.get("base_url")
                    or DEFAULT_CHATTERBOX_BASE_URL,
                    key="chatterbox_base_url_input",
                    placeholder=tr("Chatterbox Base URL Placeholder"),
                )
                _set_runtime_config(
                    "chatterbox", "base_url", (chatterbox_base_url or "").strip()
                )

                chatterbox_api_key = st.text_input(
                    tr("Chatterbox API Key"),
                    value=config.chatterbox.get("api_key", ""),
                    type="password",
                    key="chatterbox_api_key_input",
                )
                _set_runtime_config("chatterbox", "api_key", chatterbox_api_key)

                chatterbox_model = st.text_input(
                    tr("Chatterbox Model"),
                    value=config.chatterbox.get("model_id") or DEFAULT_CHATTERBOX_MODEL,
                    key="chatterbox_model_input",
                )
                _set_runtime_config(
                    "chatterbox",
                    "model_id",
                    (chatterbox_model or DEFAULT_CHATTERBOX_MODEL).strip(),
                )

                _saved_chatterbox_voices = (
                    _parse_chatterbox_voices(config.chatterbox.get("voices"))
                    or DEFAULT_CHATTERBOX_VOICES
                )
                if isinstance(_saved_chatterbox_voices, list):
                    _saved_chatterbox_voices = ", ".join(_saved_chatterbox_voices)
                chatterbox_voices = st.text_input(
                    tr("Chatterbox Voices"),
                    value=str(_saved_chatterbox_voices or ""),
                    key="chatterbox_voices_input",
                    placeholder=tr("Chatterbox Voices Placeholder"),
                )
                _set_runtime_config(
                    "chatterbox",
                    "voices",
                    _parse_chatterbox_voices(chatterbox_voices),
                )

            # Kokoro API settings section (self-hosted, OpenAI-compatible; voices listed from the server when left empty)
            if tts_mode_enabled and (
                selected_tts_server == "kokoro"
                or (voice_name and voice.is_kokoro_voice(voice_name))
            ):
                kokoro_base_url = st.text_input(
                    tr("Kokoro Base URL"),
                    value=config.kokoro.get("base_url")
                    or DEFAULT_KOKORO_BASE_URL,
                    key="kokoro_base_url_input",
                    placeholder=tr("Kokoro Base URL Placeholder"),
                )
                _set_runtime_config(
                    "kokoro", "base_url", (kokoro_base_url or "").strip()
                )

                kokoro_api_key = st.text_input(
                    tr("Kokoro API Key"),
                    value=config.kokoro.get("api_key", ""),
                    type="password",
                    key="kokoro_api_key_input",
                )
                _set_runtime_config("kokoro", "api_key", kokoro_api_key)

                kokoro_model = st.text_input(
                    tr("Kokoro Model"),
                    value=config.kokoro.get("model_id") or DEFAULT_KOKORO_MODEL,
                    key="kokoro_model_input",
                )
                _set_runtime_config(
                    "kokoro",
                    "model_id",
                    (kokoro_model or DEFAULT_KOKORO_MODEL).strip(),
                )

                _saved_kokoro_voices = (
                    _parse_chatterbox_voices(config.kokoro.get("voices"))
                    or DEFAULT_KOKORO_VOICES
                )
                if isinstance(_saved_kokoro_voices, list):
                    _saved_kokoro_voices = ", ".join(_saved_kokoro_voices)
                kokoro_voices = st.text_input(
                    tr("Kokoro Voices"),
                    value=str(_saved_kokoro_voices or ""),
                    key="kokoro_voices_input",
                    placeholder=tr("Kokoro Voices Placeholder"),
                )
                _set_runtime_config(
                    "kokoro",
                    "voices",
                    _parse_chatterbox_voices(kokoro_voices),
                )

            # The three modes only render widgets needed by the active task. Auto TTS adjusts volume and rate;
            # uploaded audio only needs file and volume; no voiceover hides unused settings.
            params.voice_name = (
                voice.NO_VOICE_NAME if voice_mode == VOICE_MODE_NONE else voice_name
            )
            params.voice_volume = 1.0
            params.voice_rate = 1.0
            uploaded_audio_file = None
            voice_volume_options = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0]
            voice_rate_options = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]

            if tts_mode_enabled:
                voice_control_cols = st.columns(2)
                with voice_control_cols[0]:
                    params.voice_volume = stable_selectbox(
                        tr("Voiceover Volume"),
                        options=voice_volume_options,
                        default_value=_saved_ui_choice(
                            "voice_volume", voice_volume_options, 1.0
                        ),
                        key="voice_volume_select",
                        format_func=lambda value: f"{int(value * 100)}%",
                        help=tr("Voiceover Volume Help"),
                    )

                with voice_control_cols[1]:
                    is_voxcpm = bool(
                        selected_tts_server == "voxcpm"
                        or (voice_name and voice.is_voxcpm_voice(voice_name))
                    )
                    params.voice_rate = stable_selectbox(
                        tr("Voiceover Speed"),
                        options=voice_rate_options,
                        default_value=_saved_ui_choice(
                            "voice_rate", voice_rate_options, 1.0
                        ),
                        key="voice_rate_select",
                        format_func=lambda value: f"{value:.1f}×",
                        help=(
                            tr("VoxCPM Speed Not Supported")
                            if is_voxcpm
                            else tr("Voiceover Speed Help")
                        ),
                        disabled=is_voxcpm,
                    )
                _set_runtime_config("ui", "voice_volume", params.voice_volume)
                _set_runtime_config("ui", "voice_rate", params.voice_rate)

                # Voice preview must sit after volume and rate controls to ensure calls use current widget values.
                _render_voice_preview(
                    params,
                    friendly_names,
                    selected_tts_server,
                    voice_name,
                )
            elif voice_mode == VOICE_MODE_UPLOAD:
                custom_audio_file_types = sorted(
                    extension.removeprefix(".") for extension in CUSTOM_AUDIO_EXTENSIONS
                )
                uploaded_audio_file = st.file_uploader(
                    tr("Upload Voiceover File"),
                    type=custom_audio_file_types
                    + [file_type.upper() for file_type in custom_audio_file_types],
                    accept_multiple_files=False,
                    key="custom_audio_file_uploader",
                    help=tr("Upload Voiceover File Help"),
                )
                params.voice_volume = stable_selectbox(
                    tr("Voiceover Volume"),
                    options=voice_volume_options,
                    default_value=_saved_ui_choice(
                        "voice_volume", voice_volume_options, 1.0
                    ),
                    key="voice_volume_select",
                    format_func=lambda value: f"{int(value * 100)}%",
                    help=tr("Voiceover Volume Help"),
                )
                _set_runtime_config("ui", "voice_volume", params.voice_volume)
                if uploaded_audio_file:
                    st.audio(uploaded_audio_file, format="audio/mp3")
                    st.info(
                        tr(
                            "Custom audio will be used directly. TTS synthesis will be skipped for this task."
                        )
                    )
            uploaded_bgm_file = _render_background_music_settings(
                params,
                elevenlabs_api_key_rendered=elevenlabs_api_key_rendered,
            )
    return uploaded_audio_file, uploaded_bgm_file, voice_mode


def _render_subtitle_settings(panel, params):
    """Render subtitle settings and update generation parameters."""
    with panel:
        with st.container(border=True):
            st.write(tr("Subtitle Settings"))
            st.session_state.setdefault(
                "subtitle_enabled_checkbox",
                _saved_ui_bool(
                    "subtitle_enabled",
                    DEFAULT_SUBTITLE_SETTINGS["subtitle_enabled"],
                ),
            )
            params.subtitle_enabled = st.checkbox(
                tr("Enable Subtitles"),
                key="subtitle_enabled_checkbox",
            )
            _set_runtime_config("ui", "subtitle_enabled", params.subtitle_enabled)
            subtitle_settings_disabled = not params.subtitle_enabled
            font_names = get_all_fonts()
            saved_font_name = config.ui.get(
                "font_name", DEFAULT_SUBTITLE_SETTINGS["font_name"]
            )
            saved_font_name_index = 0
            if saved_font_name in font_names:
                saved_font_name_index = font_names.index(saved_font_name)
            params.font_name = stable_selectbox(
                tr("Font"),
                options=font_names,
                default_value=font_names[saved_font_name_index] if font_names else "",
                key="font_name_select",
                disabled=subtitle_settings_disabled,
            )
            _set_runtime_config("ui", "font_name", params.font_name)

            subtitle_positions = [
                (tr("Top"), "top"),
                (tr("Center"), "center"),
                (tr("Bottom"), "bottom"),
                (tr("2/3 from Bottom"), "two_thirds_bottom"),
                (tr("Custom"), "custom"),
            ]
            saved_subtitle_position = config.ui.get(
                "subtitle_position", DEFAULT_SUBTITLE_SETTINGS["subtitle_position"]
            )
            saved_position_index = 2
            for i, (_, pos_value) in enumerate(subtitle_positions):
                if pos_value == saved_subtitle_position:
                    saved_position_index = i
                    break
            selected_subtitle_position = stable_selectbox(
                tr("Position"),
                options=[value for _, value in subtitle_positions],
                default_value=subtitle_positions[saved_position_index][1],
                key="subtitle_position_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in subtitle_positions
                ).get(value, value),
                disabled=subtitle_settings_disabled,
            )
            params.subtitle_position = selected_subtitle_position
            _set_runtime_config("ui", "subtitle_position", params.subtitle_position)

            # Subtitle Display Mode (Sentence vs Single Word)
            subtitle_display_modes = [
                (tr("Sentence by Sentence"), "sentence"),
                (tr("Single Word (Word by Word)"), "word_by_word"),
            ]
            saved_display_mode = config.ui.get(
                "subtitle_display_mode",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_display_mode"],
            )
            saved_mode_idx = 0
            for i, (_, mode_val) in enumerate(subtitle_display_modes):
                if mode_val == saved_display_mode:
                    saved_mode_idx = i
                    break
            selected_display_mode = stable_selectbox(
                tr("Display Mode"),
                options=[val for _, val in subtitle_display_modes],
                default_value=subtitle_display_modes[saved_mode_idx][1],
                key="subtitle_display_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in subtitle_display_modes
                ).get(value, value),
                help=tr("Word-by-word Timing Help"),
                disabled=subtitle_settings_disabled,
            )
            params.subtitle_display_mode = selected_display_mode
            _set_runtime_config(
                "ui", "subtitle_display_mode", params.subtitle_display_mode
            )

            # Subtitle Animation (None vs Pop Spring)
            subtitle_animations = [
                (tr("None (Animation)"), "none"),
                (tr("Pop Up (Spring)"), "pop_spring"),
            ]
            saved_anim = config.ui.get(
                "subtitle_animation",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_animation"],
            )
            saved_anim_idx = 0
            for i, (_, anim_val) in enumerate(subtitle_animations):
                if anim_val == saved_anim:
                    saved_anim_idx = i
                    break
            selected_anim = stable_selectbox(
                tr("Subtitle Animation"),
                options=[val for _, val in subtitle_animations],
                default_value=subtitle_animations[saved_anim_idx][1],
                key="subtitle_animation_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in subtitle_animations
                ).get(value, value),
                disabled=subtitle_settings_disabled,
            )
            params.subtitle_animation = selected_anim
            _set_runtime_config(
                "ui", "subtitle_animation", params.subtitle_animation
            )

            if params.subtitle_position == "custom":
                saved_custom_position = config.ui.get(
                    "custom_position", DEFAULT_SUBTITLE_SETTINGS["custom_position"]
                )
                st.session_state.setdefault(
                    "custom_position_input", str(saved_custom_position)
                )
                custom_position = st.text_input(
                    tr("Custom Position (% from top)"),
                    key="custom_position_input",
                    disabled=subtitle_settings_disabled,
                )
                try:
                    params.custom_position = float(custom_position)
                    if params.custom_position < 0 or params.custom_position > 100:
                        st.error(tr("Please enter a value between 0 and 100"))
                    else:
                        _set_runtime_config(
                            "ui", "custom_position", params.custom_position
                        )
                except ValueError:
                    st.error(tr("Please enter a valid number"))

            # Color labels in non-Chinese languages are typically longer than Chinese. Reserve sufficient width
            # for color pickers to prevent label wrapping while keeping ample slider room.
            font_cols = st.columns([0.42, 0.58])
            with font_cols[0]:
                saved_text_fore_color = config.ui.get(
                    "text_fore_color", DEFAULT_SUBTITLE_SETTINGS["text_fore_color"]
                )
                st.session_state.setdefault("font_color_picker", saved_text_fore_color)
                params.text_fore_color = st.color_picker(
                    tr("Font Color"),
                    key="font_color_picker",
                    disabled=subtitle_settings_disabled,
                )
                _set_runtime_config("ui", "text_fore_color", params.text_fore_color)

            with font_cols[1]:
                saved_font_size = config.ui.get(
                    "font_size", DEFAULT_SUBTITLE_SETTINGS["font_size"]
                )
                st.session_state.setdefault("font_size_slider", saved_font_size)
                params.font_size = st.slider(
                    tr("Font Size"),
                    30,
                    100,
                    key="font_size_slider",
                    disabled=subtitle_settings_disabled,
                )
                _set_runtime_config("ui", "font_size", params.font_size)

            stroke_cols = st.columns([0.42, 0.58])
            with stroke_cols[0]:
                st.session_state.setdefault(
                    "stroke_color_picker",
                    _saved_ui_color(
                        "stroke_color", DEFAULT_SUBTITLE_SETTINGS["stroke_color"]
                    ),
                )
                params.stroke_color = st.color_picker(
                    tr("Stroke Color"),
                    key="stroke_color_picker",
                    disabled=subtitle_settings_disabled,
                )
                _set_runtime_config("ui", "stroke_color", params.stroke_color)
            with stroke_cols[1]:
                st.session_state.setdefault(
                    "stroke_width_slider",
                    _saved_ui_number(
                        "stroke_width",
                        DEFAULT_SUBTITLE_SETTINGS["stroke_width"],
                        0.0,
                        10.0,
                    ),
                )
                params.stroke_width = st.slider(
                    tr("Stroke Width"),
                    0.0,
                    10.0,
                    key="stroke_width_slider",
                    disabled=subtitle_settings_disabled,
                )
                _set_runtime_config("ui", "stroke_width", params.stroke_width)

            # Background toggle localized names are generally longer than color labels; allocate slightly more space.
            subtitle_bg_cols = st.columns([0.55, 0.45])
            saved_subtitle_background_enabled = config.ui.get(
                "subtitle_background_enabled",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_background_enabled"],
            )
            st.session_state.setdefault(
                "subtitle_background_enabled_checkbox",
                saved_subtitle_background_enabled,
            )
            with subtitle_bg_cols[0]:
                subtitle_background_enabled = st.checkbox(
                    tr("Enable Subtitle Background"),
                    key="subtitle_background_enabled_checkbox",
                    disabled=subtitle_settings_disabled,
                )
            _set_runtime_config(
                "ui",
                "subtitle_background_enabled",
                subtitle_background_enabled,
            )

            # Background color and corner radius depend on the subtitle background switch. Child widgets remain visible
            # on the page and are uniformly disabled when the parent switch is off, avoiding layout jumps from appearing/disappearing widgets.
            # Color values remain in UI config, resuming prior choices when re-enabled; generation params receive False to disable rendering.
            saved_subtitle_background_color = config.ui.get(
                "subtitle_background_color",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_background_color"],
            )
            st.session_state.setdefault(
                "subtitle_background_color_picker",
                saved_subtitle_background_color,
            )
            with subtitle_bg_cols[1]:
                selected_subtitle_background_color = st.color_picker(
                    tr("Subtitle Background Color"),
                    key="subtitle_background_color_picker",
                    disabled=subtitle_settings_disabled
                    or not subtitle_background_enabled,
                )
            _set_runtime_config(
                "ui",
                "subtitle_background_color",
                selected_subtitle_background_color,
            )
            params.text_background_color = (
                selected_subtitle_background_color
                if subtitle_background_enabled
                else False
            )

            saved_rounded_subtitle_background = config.ui.get(
                "rounded_subtitle_background",
                DEFAULT_SUBTITLE_SETTINGS["rounded_subtitle_background"],
            )
            # When background is disabled, rounded backgrounds have no rendered base color. Disable widget while retaining config
            # so users can keep rounded corner preferences upon re-enabling subtitle background.
            rounded_background_disabled = (
                subtitle_settings_disabled or not subtitle_background_enabled
            )
            st.session_state.setdefault(
                "rounded_subtitle_background_checkbox",
                saved_rounded_subtitle_background,
            )
            selected_rounded_subtitle_background = st.checkbox(
                tr("Rounded Subtitle Background"),
                help=tr("Rounded Subtitle Background Help"),
                disabled=rounded_background_disabled,
                key="rounded_subtitle_background_checkbox",
            )
            params.rounded_subtitle_background = (
                selected_rounded_subtitle_background
                if subtitle_background_enabled
                else False
            )
            if not subtitle_settings_disabled and subtitle_background_enabled:
                _set_runtime_config(
                    "ui",
                    "rounded_subtitle_background",
                    selected_rounded_subtitle_background,
                )

            if video.subtitle_colors_are_indistinguishable(params):
                # Monochromatic configs remain valid user choices; prompt inline in subtitle settings without blocking generation,
                # allowing users to decide based on visual requirements.
                st.warning(tr("Subtitle Colors Are Indistinguishable"))

            subtitle_preview_text = params.video_script or params.video_subject
            selected_font_path = os.path.join(font_dir, params.font_name)
            if (
                params.subtitle_enabled
                and subtitle_preview_text
                and not video.subtitle_font_supports_text(
                    selected_font_path, subtitle_preview_text
                )
            ):
                st.warning(tr("Subtitle Font Does Not Support Text"))

            if st.button(
                tr("Restore Default Subtitle Settings"),
                key="restore_default_subtitle_settings",
                icon=":material/restart_alt:",
                on_click=reset_subtitle_settings,
                use_container_width=True,
            ):
                st.toast(tr("Default Subtitle Settings Restored"))


def _stage_task_audio(audio_path, audio_bytes):
    """Publish task-local audio only after the complete write succeeds."""
    descriptor = None
    staged_path = None
    try:
        descriptor, staged_path = tempfile.mkstemp(
            dir=os.path.dirname(audio_path), prefix=".task-audio-", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as file:
            descriptor = None  # The file context now owns the descriptor.
            file.write(audio_bytes)
        os.replace(staged_path, audio_path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if staged_path is not None:
            try:
                os.unlink(staged_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(f"failed to remove staged task audio: {exc}")


def _render_generation_controls(
    params, uploaded_files, uploaded_audio_file, uploaded_bgm_file, voice_mode
):
    """
    Validate generation dependencies, submit task, and render logs and final video outputs.

    Returns whether a new task was successfully submitted during this page run. Non-blocking save
    was requested prior to submission, allowing callers to skip redundant requests at page end.
    Main script must terminate promptly so periodic fragments can refresh progress and task logs continuously.
    """
    restore_upload_requirements = st.session_state.get(
        "task_restore_upload_requirements", {}
    )
    has_local_materials = bool(
        uploaded_files or st.session_state.get("local_video_materials", [])
    )
    has_custom_audio = bool(uploaded_audio_file)
    unmet_restore_requirements = _get_unmet_restore_upload_requirements(
        restore_upload_requirements,
        video_source=params.video_source,
        voice_name=params.voice_name or "",
        has_local_materials=has_local_materials,
        has_custom_audio=has_custom_audio,
        voice_mode=voice_mode,
    )
    if "local_materials" in unmet_restore_requirements:
        st.warning(tr("Task Restore Local Materials Warning"))
    if "custom_audio" in unmet_restore_requirements:
        st.warning(tr("Task Restore Custom Audio Warning"))
    if restore_upload_requirements and not unmet_restore_requirements:
        # Users re-uploaded files or switched material source/voice. Historical upload dependencies
        # have been explicitly addressed; clear flags to avoid showing old hints during subsequent generations.
        st.session_state.pop("task_restore_upload_requirements", None)

    _render_settings_transfer(params)

    start_button = st.button(
        tr("Generate Video"),
        use_container_width=True,
        type="primary",
        key="generate_video_button",
        on_click=_prepare_generation_task,
    )
    render_onboarding_tour()
    if start_button:
        _save_runtime_config()
        task_id = st.session_state.get("pending_generation_task_id") or str(uuid4())
        _add_active_generation_task(
            task_id,
            subject=params.video_subject or params.video_script or task_id,
        )
        if not params.video_subject and not params.video_script:
            _remove_active_generation_task(task_id)
            st.error(tr("Video Script and Subject Cannot Both Be Empty"))
            st.stop()

        voxcpm_reference_audio = None
        voxcpm_prompt_audio = None
        voxcpm_prompt_text = ""
        if voice.is_voxcpm_voice(params.voice_name or ""):
            reference_audio_error = st.session_state.get(
                VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY
            )
            if reference_audio_error:
                _remove_active_generation_task(task_id)
                st.error(
                    tr("VoxCPM Reference Audio Invalid").format(
                        error=reference_audio_error
                    )
                )
                st.stop()
            voxcpm_reference_audio = _get_voxcpm_reference_audio()
            prompt_validation_error = _get_voxcpm_prompt_validation_error()
            if prompt_validation_error:
                _remove_active_generation_task(task_id)
                st.error(
                    tr("VoxCPM Prompt Invalid").format(
                        error=prompt_validation_error
                    )
                )
                st.stop()
            voxcpm_prompt_audio = _get_voxcpm_effective_prompt_audio()
            voxcpm_prompt_text = _get_voxcpm_prompt_text()

        if params.video_source not in [
            "pexels",
            "pixabay",
            "coverr",
            "wavespeed",
            "volcengine_seedance",
            "ofox",
            "metaso_minimax",
            "muapi",
            "loomloom",
            "openai_image",
            "local",
        ]:
            _remove_active_generation_task(task_id)
            st.error(tr("Please Select a Valid Video Source"))
            st.stop()

        if params.video_source == "pexels" and not config.app.get(
            "pexels_api_keys", ""
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Pexels API Key"))
            st.stop()

        if params.video_source == "pixabay" and not config.app.get(
            "pixabay_api_keys", ""
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Pixabay API Key"))
            st.stop()

        if params.video_source == "coverr" and not config.app.get(
            "coverr_api_keys", ""
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Coverr API Key"))
            st.stop()

        if params.video_source == "wavespeed" and not config.app.get(
            "wavespeed_api_keys", ""
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the WaveSpeed API Key"))
            st.stop()

        if params.video_source == "wavespeed" and not st.session_state.get(
            "wavespeed_confirm_charge", False
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Confirm WaveSpeed Charge Required"))
            st.stop()

        if params.video_source == "volcengine_seedance" and not (
            volcengine_seedance.is_enabled(
                config.snapshot_config_with_pending(config.app)
            )
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Volcano Engine Ark API Key"))
            st.stop()

        if params.video_source == "volcengine_seedance" and not st.session_state.get(
            "volcengine_seedance_confirm_charge", False
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Confirm Volcano Engine Seedance Charge Required"))
            st.stop()

        if params.video_source == "ofox" and not (
            ofox.is_enabled(config.snapshot_config_with_pending(config.app))
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the OFox API Key"))
            st.stop()

        if params.video_source == "ofox" and not st.session_state.get(
            "ofox_confirm_charge", False
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Confirm OFox Charge Required"))
            st.stop()

        if params.video_source == "metaso_minimax" and not (
            metaso_minimax.is_enabled(
                config.snapshot_config_with_pending(config.app)
            )
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Metaso MiniMax API Key"))
            st.stop()

        if params.video_source == "metaso_minimax" and not st.session_state.get(
            "metaso_minimax_confirm_charge", False
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Confirm Metaso MiniMax Charge Required"))
            st.stop()

        if params.video_source == "muapi" and not (
            muapi.is_enabled(config.snapshot_config_with_pending(config.app))
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the MuAPI API Key"))
            st.stop()

        if params.video_source == "muapi" and not st.session_state.get(
            "muapi_confirm_charge", False
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Confirm MuAPI Charge Required"))
            st.stop()

        if params.video_source == "openai_image" and not material.is_openai_image_enabled(
            config.snapshot_config_with_pending(config.app)
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Configure the OpenAI Image Source"))
            st.stop()

        loomloom_video_request = None
        if params.video_source == "loomloom":
            current_batch, current_signature = _current_loomloom_video_quote_context(
                params
            )
            quoted_batch = st.session_state.get("loomloom_video_batch")
            quote_result = st.session_state.get("loomloom_video_quote")
            quote_is_current = bool(
                current_batch is not None
                and isinstance(quoted_batch, loomloom.LoomLoomVideoBatch)
                and quote_result is not None
                and st.session_state.get("loomloom_video_input_signature")
                == current_signature
            )
            if not quote_is_current:
                _remove_active_generation_task(task_id)
                st.error(tr("AI Video Quote Required"))
                st.stop()
            if not st.session_state.get("loomloom_video_confirm_charge", False):
                _remove_active_generation_task(task_id)
                st.error(tr("Confirm AI Video Charge Required"))
                st.stop()
            try:
                video_backend = _create_loomloom_video_backend()
                loomloom_video_request = loomloom.LoomLoomConfirmedVideoRequest(
                    settings=video_backend.settings,
                    batch=current_batch,
                    listing_version_id=quote_result.listing_version_id,
                    client_request_id=st.session_state[
                        "loomloom_video_client_request_id"
                    ],
                )
                loomloom_video_request.validate()
            except (loomloom.LoomLoomError, ValueError) as exc:
                _remove_active_generation_task(task_id)
                st.error(str(exc))
                st.stop()

        if (
            params.bgm_type == "sonilo"
            and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
            and not sonilo_service.is_enabled()
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Sonilo API Key Required"))
            st.stop()

        if (
            params.bgm_type == "elevenlabs"
            and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
            and not elevenlabs_music_service.is_enabled()
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("ElevenLabs API Key Required"))
            st.stop()

        if params.video_source == "local" and not has_local_materials:
            # Proceeding with empty local materials produces TTS/subtitles before failing during preprocessing.
            # Intercepting before task launch prevents wasteful API calls and intermediate files.
            _remove_active_generation_task(task_id)
            st.error(tr("Please Upload Local Materials First"))
            st.stop()

        if voice_mode == VOICE_MODE_UPLOAD and not uploaded_audio_file:
            # Uploaded audio is an explicit user choice; missing files must not silently fall back to TTS.
            # Intercept before task launch to prevent generating videos inconsistent with user selections.
            _remove_active_generation_task(task_id)
            st.error(tr("Please Upload Voiceover File First"))
            st.stop()

        if "custom_audio" in unmet_restore_requirements:
            # Historical custom audio cannot be repopulated automatically. Block silent fallback to TTS
            # if user has not re-uploaded or switched voices, avoiding regenerating videos with mismatched audio.
            _remove_active_generation_task(task_id)
            st.error(tr("Task Restore Custom Audio Warning"))
            st.stop()

        if uploaded_bgm_file and bgm_service.should_use_bgm(
            params.bgm_type, params.bgm_volume
        ):
            try:
                saved_bgm_name = bgm_service.save_bgm_upload(
                    uploaded_bgm_file.name, uploaded_bgm_file
                )
            except bgm_service.BgmUploadError as exc:
                _remove_active_generation_task(task_id)
                logger.warning(f"WebUI background music upload rejected: {str(exc)}")
                st.error(tr("Invalid Background Music"))
                st.stop()
            except bgm_service.BgmServiceError as exc:
                _remove_active_generation_task(task_id)
                logger.error(f"WebUI background music upload failed: {str(exc)}")
                st.error(tr("Background Music Validation Failed"))
                st.stop()
            # Write only filename to task params upon successful save. Video service resolves files within
            # two BGM whitelist directories, avoiding persisting or displaying absolute server paths to users.
            params.bgm_file = saved_bgm_name
        elif uploaded_bgm_file:
            # At 0 volume, video service omits BGM; skip persisting previewed upload files to storage.
            # Users can persist by clicking Generate again after raising volume.
            params.bgm_file = ""

        if uploaded_audio_file:
            try:
                task_dir = utils.task_dir(task_id)
                custom_audio_path = _build_uploaded_file_path(
                    uploaded_audio_file,
                    task_dir,
                    CUSTOM_AUDIO_EXTENSIONS,
                    "custom-audio",
                )
                # Voiceover uploads previously bypassed the same full-decode
                # and size checks used for background-music uploads. Validate
                # before storing or scheduling the generation task.
                bgm_service.validate_bgm_upload(
                    uploaded_audio_file.name, uploaded_audio_file
                )
                _stage_task_audio(custom_audio_path, uploaded_audio_file.getbuffer())
            except OSError as exc:
                _remove_active_generation_task(task_id)
                logger.error(f"failed to persist uploaded task audio: {exc}")
                st.error(tr("Video Generation Failed"))
                st.stop()
            except bgm_service.BgmUploadError as exc:
                _remove_active_generation_task(task_id)
                logger.warning(f"WebUI custom audio upload rejected: {exc}")
                st.error(str(exc))
                st.stop()
            except bgm_service.BgmServiceError as exc:
                _remove_active_generation_task(task_id)
                logger.error(f"WebUI custom audio validation failed: {exc}")
                st.error(str(exc))
                st.stop()
            except ValueError:
                _remove_active_generation_task(task_id)
                st.error(tr("Unsupported Upload File Type"))
                st.stop()
            params.custom_audio_file = custom_audio_path

        if uploaded_files:
            # On each re-upload, use currently selected materials, preventing old materials from being repeatedly appended.
            try:
                params.video_materials, persisted_local_materials = (
                    _save_uploaded_local_materials(uploaded_files)
                )
            except material_upload_service.MaterialUploadError as exc:
                _remove_active_generation_task(task_id)
                logger.warning(f"WebUI local material upload rejected: {exc}")
                st.error(str(exc))
                st.stop()
            except material_upload_service.MaterialServiceError as exc:
                _remove_active_generation_task(task_id)
                logger.error(f"WebUI local material upload failed: {exc}")
                st.error(str(exc))
                st.stop()
            # Write uploaded local video materials into session for reuse when updating script copy only.
            st.session_state["local_video_materials"] = persisted_local_materials
        elif (
            params.video_source == "local" and st.session_state["local_video_materials"]
        ):
            # Reuse the most recent local material list saved to disk when no new files are uploaded.
            params.video_materials = []
            for material_entry in st.session_state["local_video_materials"]:
                m = MaterialInfo()
                m.provider = material_entry.get("provider", "local")
                m.url = material_entry.get("url", "")
                m.duration = material_entry.get("duration", 0)
                if m.url:
                    params.video_materials.append(m)

        reusable_voice_preview = _get_reusable_full_voice_preview(
            params,
            voice_mode,
        )
        if reusable_voice_preview:
            # Preview cache only exists within current Streamlit session. Write audio into target task directory before submission;
            # background threads subsequently read task-owned files only. Page reruns, browser closures,
            # or previews of other voices will not affect queued generation tasks.
            try:
                preview_audio_file = os.path.join(
                    utils.task_dir(task_id),
                    "audio.mp3",
                )
                _stage_task_audio(preview_audio_file, reusable_voice_preview["audio_bytes"])
            except OSError as exc:
                _remove_active_generation_task(task_id)
                logger.error(f"failed to persist preview task audio: {exc}")
                st.error(tr("Video Generation Failed"))
                st.stop()
            reusable_voice_preview.pop("audio_bytes")
            reusable_voice_preview["audio_file"] = preview_audio_file
            logger.info(
                f"reuse full voice preview for task: "
                f"task_id={task_id}, duration={reusable_voice_preview['duration']:.2f}s"
            )

        try:
            st.toast(tr("Generating Video"))
            logger.info(tr("Start Generating Video"))
            logger.info(utils.to_json(params))
            webui_task.submit_generation(
                task_id=task_id,
                params=params,
                capture_logs=not config.ui.get("hide_log", False),
                voice_preview=reusable_voice_preview,
                loomloom_video_request=loomloom_video_request,
                voxcpm_reference_audio=voxcpm_reference_audio,
                voxcpm_prompt_audio=voxcpm_prompt_audio,
                voxcpm_prompt_text=voxcpm_prompt_text,
            )
            if loomloom_video_request is not None:
                # Each quote allows only one submission. Background requests include stable idempotent IDs;
                # clear page quotes upon submission, requiring requoting and confirmation for subsequent runs.
                st.session_state["loomloom_video_batch"] = None
                st.session_state["loomloom_video_quote"] = None
                st.session_state["loomloom_video_input_signature"] = ""
                st.session_state["loomloom_video_client_request_id"] = ""
        except Exception:
            _remove_active_generation_task(task_id)
            st.error(tr("Video Generation Failed"))
            st.stop()

        st.session_state["current_generation_task_id"] = task_id
        logger.info(f"WebUI generation task submitted: task_id={task_id}")

    _render_current_generation_task()
    return start_button


def _render_application():
    """Render top bar, dialogs, generation form, and task results in fixed sequence."""
    _render_top_bar()

    if st.session_state.get("settings_dialog_open", False):
        _render_settings_dialog()

    if _apply_pending_settings_preset():
        st.success(tr("Settings Preset Imported"))

    restore_applied = _apply_pending_task_restore()
    restore_candidate_id = st.session_state.get("task_restore_candidate_id")
    if restore_candidate_id:
        _render_task_restore_dialog(restore_candidate_id)
    restore_succeeded = st.session_state.pop("task_restore_succeeded", False)
    if restore_applied or restore_succeeded:
        st.success(tr("Task Configuration Loaded"))

    with st.container(key="main_settings_grid"):
        panel = st.columns(4)
    left_panel = panel[0]
    middle_panel = panel[1]
    audio_panel = panel[2]
    right_panel = panel[3]

    params = VideoParams(video_subject="")
    params.match_materials_to_script = bool(
        st.session_state.get("match_materials_to_script", False)
    )
    _render_script_settings(left_panel, params)

    uploaded_files = _render_video_settings(middle_panel, params)
    uploaded_audio_file, uploaded_bgm_file, voice_mode = _render_audio_settings(
        audio_panel, params
    )

    _render_subtitle_settings(right_panel, params)

    generation_submitted = _render_generation_controls(
        params,
        uploaded_files,
        uploaded_audio_file,
        uploaded_bgm_file,
        voice_mode,
    )

    # The generation branch requests a save before starting worker threads. Regular widget interactions continue
    # requesting non-blocking saves; if background tasks hold the config lock, latest values are applied upon task completion.
    if not generation_submitted:
        _save_runtime_config()


_render_application()
