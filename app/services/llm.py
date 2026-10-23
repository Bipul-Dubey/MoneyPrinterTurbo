import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from time import perf_counter
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider
from app.utils import utils
from app.utils.request_logging import trace_sdk_call

_max_retries = 5
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
# "Test model connection" sends only a minimal request. The OpenAI SDK defaults to a 600-second timeout with 2 automatic
# retries; when endpoints are unreachable (proxy, firewall, or DNS issues), the test button hangs for minutes before
# showing only "Request timed out." (see issue #1642). A bounded short timeout and disabled SDK retries are applied here
# for fast, diagnostic failures; formal script generation retains SDK defaults to avoid penalizing slow reasoning models or local Ollama cold starts.
LLM_CONNECTION_TEST_TIMEOUT_SECONDS = 30.0
LLM_CONNECTION_TEST_MAX_RETRIES = 0
# These snippets originate from OpenAI SDK APITimeoutError / APIConnectionError and underlying httpx connection exceptions;
# matching any treats the error as "endpoint unreachable".
_CONNECTION_FAILURE_MARKERS = (
    "timed out",
    "timeout",
    "connection error",
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to connect",
)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_UNOPENED_THINK_BLOCK_RE = re.compile(r"^.*</think\s*>", re.IGNORECASE | re.DOTALL)
_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\s?#@]+@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\s]+)",
    re.IGNORECASE,
)

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.
""".strip()

# Claude Code CLI defaults to a coding agent system prompt with many constraints irrelevant to copywriting,
# which can skew script and keyword generation away from instructions; therefore replace it completely on invocation.
CLAUDE_CODE_SYSTEM_PROMPT = (
    "You are a concise copywriter. Follow the user's instructions and output "
    "format exactly, and output nothing else."
)
CLAUDE_CODE_DEFAULT_TIMEOUT = 300.0
# `--tools ""` disables all built-in tools; `--safe-mode` disables CLAUDE.md, skills, hooks, plugins, MCP,
# and all user-level customizations while preserving auth, model selection, and permissions.
# Both require recent CLI versions; older versions exit with "unknown option", converted by callers into clear guidance.
CLAUDE_CODE_MIN_CLI_VERSION = "2.1.260"
# These environment variables cause the CLI to switch to API keys or third-party providers (Bedrock, Vertex, Foundry,
# Mantle, Gateway, etc.), bypassing subscription login and incurring separate charges. Rather than enumerating variables,
# filter them by category prefixes:
#   ANTHROPIC_*           API Key, Auth Token, Base URL, provider endpoints, and profiles
#   CLAUDE_CODE_USE_*     Provider toggles
#   CLAUDE_CODE_SKIP_*_AUTH  Bypass provider auth toggles
CLAUDE_CODE_CONFLICTING_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_")
CLAUDE_CODE_CONFLICTING_ENV_VARS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "CLAUDE_CODE_GATEWAY_TOKEN_FILE_DESCRIPTOR",
)
# These two categories must be retained:
#   CLAUDE_CODE_OAUTH_TOKEN is the sole subscription auth mechanism inside containers (does not match the prefix);
#   *_CONFIG_DIR designates credentials storage; removing it invalidates logged-in subscriptions.
CLAUDE_CODE_PRESERVED_ENV_VARS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_CONFIG_DIR",
    "CLAUDE_CONFIG_DIR",
)


def _is_conflicting_claude_code_env(name: str) -> bool:
    """Determine whether an environment variable would divert the CLI from subscription login to alternative auth methods."""
    if name in CLAUDE_CODE_PRESERVED_ENV_VARS:
        return False
    if name in CLAUDE_CODE_CONFLICTING_ENV_VARS:
        return True
    if name.startswith(CLAUDE_CODE_CONFLICTING_ENV_PREFIXES):
        return True
    return name.startswith("CLAUDE_CODE_SKIP_") and name.endswith("_AUTH")


def coerce_claude_code_timeout(value, config_key: str = "claude_code_timeout"):
    """
    Parse configured timeout into a positive finite number of seconds.

    TOML values may appear as `claude_code_timeout = 300` (int/float) or `"300"` (str), so `strip()` cannot be called directly.
    Rejects nan and inf values that would cause `subprocess.run(timeout=...)` to hang indefinitely.
    """
    if value is None:
        return CLAUDE_CODE_DEFAULT_TIMEOUT

    if isinstance(value, bool):
        # bool is a subclass of int, but True seconds is clearly not the user's intended timeout.
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return CLAUDE_CODE_DEFAULT_TIMEOUT
        try:
            seconds = float(text)
        except ValueError:
            raise ValueError(
                f"{config_key} must be a number of seconds, got {value!r}"
            ) from None
    elif isinstance(value, (int, float)):
        seconds = float(value)
    else:
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if not math.isfinite(seconds):
        raise ValueError(f"{config_key} must be a finite number, got {value!r}")
    if seconds <= 0:
        raise ValueError(f"{config_key} must be greater than 0, got {value!r}")
    return seconds


def _resolve_provider_field_value(raw_value, default_value):
    """
    Fall back to Registry default only when value is unconfigured.

    Previously `raw or default_value` treated valid falsy values like 0 or False as unconfigured:
    `claude_code_timeout = 0` was silently replaced with 300, while `"0"` threw an error.
    Defaults apply only on None or empty strings, keeping validation consistent across types.
    """
    if raw_value is None:
        return default_value
    if isinstance(raw_value, str) and not raw_value.strip():
        return default_value
    return raw_value


def build_claude_code_env(base_env=None):
    """
    Construct a subprocess environment relying strictly on subscription login.

    Returns (env_dict, stripped_vars). Strips variables that alter auth modes or providers;
    `CLAUDE_CODE_OAUTH_TOKEN` must be kept: containers have no keychain, so CLI relies on it for subscription auth.
    """
    env = dict(os.environ if base_env is None else base_env)
    removed = sorted(name for name in env if _is_conflicting_claude_code_env(name))
    for name in removed:
        env.pop(name, None)
    return env, removed


def _normalize_text_response(content, llm_provider: str) -> str:
    # Different LLM SDKs may return None, empty strings, or non-string objects on errors or interception.
    # Perform uniform fallback validation here to prevent AttributeError like `NoneType` on `.replace()`.
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    # Reasoning models such as MiniMax M3 or DeepSeek R1 may wrap internal chain-of-thought inside `<think>...</think>`.
    # Video scripts and keywords require only final readable text; without centralized sanitization, WebUI, subtitles,
    # and voiceover will treat thinking processes as narration content.
    content = _THINK_BLOCK_RE.sub("", content)
    # Chat templates for models like DeepSeek R1 and QwQ insert an opening `<think>` into the prompt,
    # causing gateways like vLLM/Ollama to return `reasoning...</think>answer`. Content preceding remaining closing tags
    # after paired cleanup is also reasoning output and must be stripped from the script.
    content = _UNOPENED_THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    # Prior ``strip()`` removes leading and trailing whitespace. Single and double newlines within body text
    # must be preserved: script generation relies on double newlines for paragraphs, and subtitle processing reads line-by-line.
    return content


def _sanitize_error_message(error: object) -> str:
    """
    Sanitize error messages returned to WebUI/API, preventing credential leakage in custom base_urls.

    Some OpenAI-compatible SDKs echo raw request URLs into exception strings. If users configure
    `https://user:pass@example.com/v1` for proxy gateways, returning `str(e)` directly exposes credentials
    to the UI, API callers, or logs. Only the error message is sanitized here without altering the actual request URL.
    """
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1***", message)
    return message


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI-compatible endpoints may return responses lacking choices or with empty choices/message/content under error conditions.
    # Uniform structural validation prevents low-level exceptions like `NoneType is not subscriptable`.
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _get_response_field(value, key: str):
    """Safely read fields from both dict and SDK response objects."""
    if isinstance(value, dict):
        return value.get(key)

    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _sdk_transport_options(request_timeout, max_retries) -> dict:
    """Convert optional timeout and retry constraints into OpenAI SDK client constructor kwargs.

    Returns an empty dict when both arguments are None, retaining SDK defaults (600s timeout, 2 auto-retries)
    for routine script generation; tighter budgets apply only when callers like connection tests explicitly pass bounded values.
    """
    options = {}
    if request_timeout is not None:
        options["timeout"] = request_timeout
    if max_retries is not None:
        options["max_retries"] = max_retries
    return options


def _explain_connection_test_failure(error_message: str) -> str:
    """Augment timeout and connection failures with troubleshooting guidance rather than returning raw SDK text.

    In issue #1642, the user's API was operational, but connectivity issues (proxy, firewall, DNS) failed the request,
    while "Request timed out." offered no actionable clues. Other errors (auth, quota) are returned verbatim.
    """
    normalized = (error_message or "").lower()
    if not any(marker in normalized for marker in _CONNECTION_FAILURE_MARKERS):
        return error_message
    return (
        f"{error_message} (the request did not complete within "
        f"{LLM_CONNECTION_TEST_TIMEOUT_SECONDS:.0f}s). Please check that the "
        "configured base URL is reachable from this machine, and that no proxy "
        "or firewall is blocking the request."
    )


def _extract_qwen_generation_text(response) -> str:
    """
    Extract text content from DashScope Generation responses.

    When Qwen is called with `messages`, it returns a chat payload: `output.choices[0].message.content`;
    only legacy completion mode returns `output.text`. Both formats are supported to prevent AttributeError
    when `output.text` is None.
    """
    output = _get_response_field(response, "output")
    choices = _get_response_field(output, "choices") if output else None
    if choices is not None:
        if not choices:
            logger.warning("Qwen returned an empty choices list")
            raise ValueError("[qwen] returned empty choices")

        first_choice = choices[0]
        message = _get_response_field(first_choice, "message")
        content = _get_response_field(message, "content") if message else None
        if content is not None:
            return _normalize_text_response(content, "qwen")

    text = _get_response_field(output, "text") if output else None
    return _normalize_text_response(text, "qwen")


def _generate_response(
    prompt: str,
    app_config=None,
    request_timeout: float | None = None,
    max_retries: int | None = None,
) -> str:
    sdk_client = None
    sdk_stream = None
    try:
        # The WebUI allows users to prepare their next script during video generation. Callers can supply a config snapshot
        # captured at submission time, ensuring retries do not shift to a different provider, base URL, or model if settings change.
        runtime_app_config = app_config if app_config is not None else config.app
        llm_provider = str(
            runtime_app_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
        ).lower()
        provider = get_llm_provider(llm_provider)
        if provider is None:
            raise ValueError(f"{llm_provider}: unsupported llm provider")

        logger.info(f"llm provider: {llm_provider}")
        api_key = runtime_app_config.get(provider.config_key("api_key"), "")
        configured_model = runtime_app_config.get(provider.config_key("model_name"), "")
        model_name = provider.resolve_model_name(configured_model)
        if configured_model and model_name != configured_model:
            logger.warning(
                f"{llm_provider} model '{configured_model}' is deprecated, "
                f"fallback to '{model_name}'"
            )
        configured_base_url = runtime_app_config.get(
            provider.config_key("base_url"), ""
        )
        base_url = provider.resolve_base_url(configured_base_url)
        if configured_base_url and configured_base_url.strip().rstrip("/") in {
            url.rstrip("/") for url in provider.deprecated_base_urls
        }:
            logger.warning(
                f"{llm_provider} base URL '{configured_base_url}' is deprecated, "
                f"fallback to '{base_url}'"
            )
        adapter = provider.adapter
        api_version = ""

        # Ollama default URL depends on containerized vs host execution and cannot be stored statically in Registry;
        # Registry governs models and required fields, while environment differences are resolved dynamically here.
        if llm_provider == "ollama":
            api_key = "ollama"
            if not base_url:
                base_url = config.get_default_ollama_base_url()

        if adapter == "azure":
            api_version = runtime_app_config.get(
                provider.config_key("api_version"), "2024-02-15-preview"
            )

        extra_values = {
            field.config_suffix: _resolve_provider_field_value(
                runtime_app_config.get(provider.config_key(field.config_suffix)),
                field.default_value,
            )
            for field in provider.extra_fields
        }

        if provider.requires_api_key and not api_key:
            raise ValueError(
                f"{llm_provider}: api_key is not set, please set it in the config.toml file."
            )
        if provider.requires_model_name and not model_name:
            raise ValueError(
                f"{llm_provider}: model_name is not set, please set it in the config.toml file."
            )
        if provider.requires_base_url and not base_url:
            raise ValueError(
                f"{llm_provider}: base_url is not set, please set it in the config.toml file."
            )

        for field in provider.extra_fields:
            if field.required and not extra_values[field.config_suffix]:
                raise ValueError(
                    f"{llm_provider}: {field.config_suffix} is not set, "
                    "please set it in the config.toml file."
                )

        if adapter == "qwen":
            import dashscope
            from dashscope.api_entities.dashscope_response import GenerationResponse

            response = trace_sdk_call(
                llm_provider, base_url, dashscope.Generation.call,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
            )
            if response:
                if isinstance(response, GenerationResponse):
                    status_code = response.status_code
                    if status_code != 200:
                        raise Exception(
                            f'[{llm_provider}] returned an error response: "{response}"'
                        )

                    return _extract_qwen_generation_text(response)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}"'
                    )
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        if adapter == "gemini":
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(base_url=base_url) if base_url else None
            generation_config = types.GenerateContentConfig(
                temperature=0.5,
                top_p=1,
                top_k=1,
                max_output_tokens=2048,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                ],
            )

            try:
                # The modern google-genai SDK exposes services via a unified Client. Context managers close underlying
                # HTTP connections upon request completion, preventing socket resource leaks during frequent generation.
                with genai.Client(
                    api_key=api_key,
                    http_options=http_options,
                ) as client:
                    response = trace_sdk_call(
                        llm_provider, base_url, client.models.generate_content,
                        model=model_name,
                        contents=prompt,
                        config=generation_config,
                    )
                generated_text = response.text
            except (AttributeError, IndexError, ValueError) as e:
                logger.warning(f"gemini returned invalid response content: {str(e)}")
                raise ValueError(f"[{llm_provider}] returned invalid response content")

            return _normalize_text_response(generated_text, llm_provider)

        if adapter == "cloudflare_ai_gateway":
            account_id = extra_values["account_id"]
            gateway_id = extra_values["gateway_id"]
            # Cloudflare's recommended AI Gateway REST API is OpenAI SDK-compatible. Account ID constructs the unified endpoint,
            # and Gateway ID is selected via headers; avoids calling the legacy Workers AI `/ai/run/{model}` endpoint.
            client = sdk_client = OpenAI(
                api_key=api_key,
                base_url=(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
                ),
                default_headers={"cf-aig-gateway-id": gateway_id},
                **_sdk_transport_options(request_timeout, max_retries),
            )
            response = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "litellm":
            import litellm

            if not model_name:
                raise ValueError(
                    f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                )

            response = trace_sdk_call(
                llm_provider, base_url, litellm.completion,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                drop_params=True,
            )

            if not response:
                raise ValueError(f"[{llm_provider}] returned empty response")
            if not getattr(response, "choices", None):
                raise ValueError(f"[{llm_provider}] returned empty response")

            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "azure":
            # Azure OpenAI SDK uses `azure_endpoint` and `api_version` to build request URLs, incompatible with standard
            # OpenAI-compatible `base_url` initialization. Complete the request within the Azure branch and return immediately
            # so the client is not overridden by fallback logic, which would otherwise validate Azure credentials without using them.
            logger.info(f"requesting azure chat completion, model: {model_name}")
            client = sdk_client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
                **_sdk_transport_options(request_timeout, max_retries),
            )
            response = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        if adapter == "claude_code":
            # Claude subscriptions (Pro / Max / Team) do not issue API keys; credentials belong exclusively to the official Claude Code CLI.
            # Instead of querying the Anthropic API directly, invoke the authenticated local CLI in headless mode (`claude -p`),
            # delegating auth to the CLI while consuming its generated text output.
            configured_cli = (extra_values.get("cli_path") or "").strip() or "claude"
            cli_path = shutil.which(configured_cli)
            if not cli_path and os.path.isfile(configured_cli):
                cli_path = configured_cli
            if not cli_path:
                raise ValueError(
                    f"{llm_provider}: claude CLI not found ('{configured_cli}'), "
                    f"install it in the runtime or set "
                    f"{provider.config_key('cli_path')} in the config.toml file."
                )

            try:
                timeout_seconds = coerce_claude_code_timeout(
                    extra_values.get("timeout"), provider.config_key("timeout")
                )
            except ValueError as timeout_error:
                raise ValueError(f"{llm_provider}: {timeout_error}") from None

            # Pass prompt via stdin rather than CLI args: on Windows, npm-installed claude is claude.cmd,
            # where cmd.exe truncates arguments at the first newline, dropping multi-line prompts and subsequent flags.
            command = [
                cli_path,
                "-p",
                "--output-format",
                "json",
                "--system-prompt",
                CLAUDE_CODE_SYSTEM_PROMPT,
                # Disable all built-in tools to restrict execution strictly to text generation.
                "--tools",
                "",
                # Disable CLAUDE.md, skills, hooks, plugins, MCP, and user-level customizations;
                # authentication and model selection remain functional (cannot use --bare which disables OAuth).
                "--safe-mode",
            ]
            # When model name is empty, fall back to CLI default model, avoiding hardcoded IDs that may break over subscription tiers.
            if model_name:
                command += ["--model", model_name]

            cli_env, removed_env = build_claude_code_env()
            if removed_env:
                # Log variable names only, never their values, preventing secret exposure in logs.
                logger.warning(
                    f"{llm_provider}: ignoring conflicting environment variables "
                    f"so the subscription login is used: {', '.join(removed_env)}"
                )

            logger.info(f"invoking claude cli, model: {model_name or 'cli default'}")
            # The CLI reads CLAUDE.md and project settings from the working directory, which pollutes copywriting outputs;
            # execute inside an isolated temporary directory instead.
            with tempfile.TemporaryDirectory() as work_dir:
                try:
                    completed = subprocess.run(
                        command,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        # The CLI always emits UTF-8. Without an explicit encoding,
                        # text=True decodes with the system locale (e.g. cp1252 on
                        # non-English Windows), so every non-ASCII character reaches
                        # the script as mojibake.
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout_seconds,
                        cwd=work_dir,
                        env=cli_env,
                    )
                except subprocess.TimeoutExpired:
                    raise Exception(
                        f"[{llm_provider}] claude cli timed out after "
                        f"{timeout_seconds:.0f}s"
                    )

            # Authentication and quota exhaustion errors still return JSON (`is_error` true, `result` human-readable reason)
            # despite non-zero exit codes. Parse stdout first; fall back to exit code and stderr only if JSON is missing.
            stdout = (completed.stdout or "").strip()
            try:
                payload = json.loads(stdout) if stdout else None
            except json.JSONDecodeError:
                payload = None

            if payload is None:
                detail = (completed.stderr or stdout or "").strip()
                if "unknown option" in detail.lower():
                    raise Exception(
                        f"[{llm_provider}] the installed claude CLI does not support "
                        f"the required isolation flags; upgrade to "
                        f"{CLAUDE_CODE_MIN_CLI_VERSION} or newer: {detail[:300]}"
                    )
                if completed.returncode != 0:
                    raise Exception(
                        f"[{llm_provider}] claude cli exited with code "
                        f"{completed.returncode}: {detail[:500]}"
                    )
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{detail[:500]}"'
                )

            if payload.get("is_error") or completed.returncode != 0:
                reason = str(payload.get("result") or "").strip() or (
                    f"claude cli exited with code {completed.returncode}"
                )
                # Interactive `/login` cannot be executed inside containers; provide viable auth alternatives directly.
                if "login" in reason.lower():
                    reason += (
                        " (run `claude setup-token` on the host and pass the token "
                        "to the container as CLAUDE_CODE_OAUTH_TOKEN)"
                    )
                raise Exception(
                    f'[{llm_provider}] returned an error response: "{reason[:500]}"'
                )

            return _normalize_text_response(payload.get("result"), llm_provider)

        if adapter == "kimi_code":
            # Kimi subscriptions (Kimi Code benefits under Plus+) do not issue pay-as-you-go API keys; auth relies on OAuth device flow:
            # WebUI guides users to confirm in the browser, storing the refresh token locally; access tokens expire in 15m and refresh
            # transparently via kimi_code_oauth. Requests hit official OpenAI-compatible endpoints with quotas separate from pay-as-you-go.
            from app.services import kimi_code_oauth

            try:
                timeout_seconds = kimi_code_oauth.coerce_timeout(
                    extra_values.get("timeout")
                )
            except ValueError as timeout_error:
                raise ValueError(f"{llm_provider}: {timeout_error}") from None

            access_token = kimi_code_oauth.get_valid_access_token()
            # Authorized region follows token credentials: tokens are valid only within their issuing region. If users switch
            # API Platform post-auth, sending requests by UI selection would route China tokens to Global endpoints (yielding 401),
            # so request destinations adhere strictly to the token's origin region.
            authorized_region = kimi_code_oauth.authorized_region()
            if authorized_region:
                request_base_url = kimi_code_oauth.api_base_url(authorized_region)
                if base_url and authorized_region != kimi_code_oauth.resolve_region(
                    base_url
                ):
                    logger.warning(
                        f"kimi_code: API Platform selection differs from the "
                        f"authorized region; using '{authorized_region}' to match "
                        "the issued token"
                    )
            else:
                request_base_url = base_url or kimi_code_oauth.api_base_url(
                    kimi_code_oauth.resolve_region(base_url)
                )
            # When model name is empty, default to kimi-for-coding: dedicated coding model under subscription benefits,
            # avoiding hardcoded general model IDs that may expire with plan changes.
            request_model = model_name or "kimi-for-coding"
            logger.info(f"requesting kimi code chat completion, model: {request_model}")
            # Provider generous timeouts must be clamped to the test budget during connection tests.
            effective_timeout = timeout_seconds
            if request_timeout is not None:
                effective_timeout = min(effective_timeout, request_timeout)
            client = sdk_client = OpenAI(
                api_key=access_token,
                base_url=request_base_url,
                timeout=effective_timeout,
                # Server anti-scraping pipeline requires device identity headers matching the official CLI.
                default_headers=kimi_code_oauth.api_headers(),
                **_sdk_transport_options(None, max_retries),
            )
            response = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=request_model, messages=[{"role": "user", "content": prompt}]
            )
            if response and isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            raise Exception(
                f'[{llm_provider}] returned an invalid response: "{response}"'
            )

        if adapter == "modelscope":
            content = ""
            client = sdk_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                **_sdk_transport_options(request_timeout, max_retries),
            )
            response = sdk_stream = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},
                stream=True,
            )
            if response:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content

                if not content.strip():
                    raise ValueError("Empty content in stream response")

                return _normalize_text_response(content, llm_provider)
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        client = sdk_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            **_sdk_transport_options(request_timeout, max_retries),
        )

        response = trace_sdk_call(
            llm_provider, base_url, client.chat.completions.create,
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        if response:
            if isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            else:
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                    f"connection and try again."
                )
        else:
            raise Exception(
                f"[{llm_provider}] returned an empty response, please check your network connection and try again."
            )

    except Exception as e:
        return f"Error: {_sanitize_error_message(e)}"

    finally:
        for resource in (sdk_stream, sdk_client):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as cleanup_error:
                    logger.warning(
                        f"could not close LLM transport: {type(cleanup_error).__name__}"
                    )


def test_connection() -> tuple[bool, str, float]:
    """
    Issue a minimal request using current Provider settings to verify that the generation pipeline is functional.

    Connection tests reuse `_generate_response()`, validating API Key, Base URL, model name, and provider-specific fields
    without triggering script generation retry loops or transmitting user video topics/scripts. Requests apply a bounded short timeout
    with SDK auto-retries disabled: unreachable endpoints (proxy, firewall, DNS) fail fast with troubleshooting guidance instead of
    hanging for minutes with "Request timed out." (issue #1642). Returns (success, error_message, elapsed_seconds).
    """
    started_at = perf_counter()
    response = _generate_response(
        prompt="Reply with exactly: OK",
        request_timeout=LLM_CONNECTION_TEST_TIMEOUT_SECONDS,
        max_retries=LLM_CONNECTION_TEST_MAX_RETRIES,
    )
    elapsed = perf_counter() - started_at

    if not response:
        error_message = "LLM returned an empty response"
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    if response.startswith("Error:"):
        error_message = _explain_connection_test_failure(
            response.removeprefix("Error:").strip()
        )
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    logger.info(f"llm connection test succeeded, elapsed: {elapsed:.2f}s")
    return True, "", elapsed


def _limit_script_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API routes validate lengths via Pydantic; fallback protection here prevents WebUI or internal services
    # invoking generate_script directly from transmitting oversized prompts, guarding against token exhaustion and request failures.
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _normalize_script_paragraph_number(paragraph_number: int | None) -> int:
    try:
        value = int(paragraph_number or MIN_SCRIPT_PARAGRAPH_NUMBER)
    except (TypeError, ValueError):
        value = MIN_SCRIPT_PARAGRAPH_NUMBER

    if value < MIN_SCRIPT_PARAGRAPH_NUMBER or value > MAX_SCRIPT_PARAGRAPH_NUMBER:
        # WebUI and API enforce parameter boundaries; guard internal calls here to prevent aberrant parameters
        # from inflating LLM token expenses or yielding empty outputs.
        logger.warning(
            f"script paragraph_number is out of range and will be clamped: {value}"
        )
        return max(MIN_SCRIPT_PARAGRAPH_NUMBER, min(value, MAX_SCRIPT_PARAGRAPH_NUMBER))

    return value


def build_script_prompt(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )

    # Separate "script generation rules" and "runtime context" during concatenation. Advanced users overriding the default
    # system prompt will still retain essential per-run parameters such as video topic, language, and paragraph count.
    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    prompt += f"""

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if language:
        prompt += f"\n- language: {language}"
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    return prompt


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    app_config=None,
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )
    prompt = build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    final_script = ""
    logger.info(
        "generating video script: "
        f"subject={video_subject}, paragraph_number={paragraph_number}, "
        f"has_custom_prompt={bool(video_script_prompt.strip())}, "
        f"has_custom_system_prompt={bool(custom_system_prompt.strip())}"
    )

    def format_response(response):
        # Clean the script
        # Remove emphasis and heading markers without changing inline names
        # or musical notes such as C#, F# and G#.
        response = response.replace("*", "")
        response = re.sub(
            r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.*?)[ \t]*$",
            lambda match: re.sub(r"[ \t]+#+[ \t]*$", "", match.group(1)),
            response,
        )

        # Link labels are narration; discard destinations before stripping
        # bracketed stage directions. Common parenthesized URL components fit.
        response = re.sub(
            r"\[([^\]\n]+)\]\((?:[^()\n]|\([^()\n]*\))*\)",
            lambda match: match.group(1),
            response,
        )

        # Remove markdown syntax.  Use non-greedy .*? so each bracket/paren
        # group is removed independently; the greedy form would eat all text
        # between the first opener and the last closer on the same line.
        response = re.sub(r"\[.*?\]", "", response)
        response = re.sub(r"\(.*?\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        return "\n\n".join(paragraphs)

    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt=prompt)
            else:
                response = _generate_response(prompt=prompt, app_config=app_config)
            if isinstance(response, str) and response.startswith("Error: "):
                # _generate_response returns provider failures as text. Passing
                # that text through would make the task treat it as narration.
                raise ValueError(response)
            if response:
                candidate = format_response(response).strip()
            else:
                logging.error("gpt returned an empty response")
                candidate = ""

            # Some upstream providers may return quota errors as plain text.
            if candidate and "当日额度已消耗完" in candidate:
                raise ValueError(candidate)

            if candidate:
                final_script = candidate
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries - 1:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    if not final_script:
        logger.error("failed to generate video script after retries")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding markdown code fence from an LLM response.

    Non-OpenAI providers (Claude, Gemini, …) frequently wrap JSON output in a
    ```json … ``` fence even when asked to return raw JSON. Removing it lets the
    first json.loads() succeed instead of falling through to the regex recovery
    path (and spuriously logging a warning). Mirrors the DOTALL handling already
    used in _parse_social_metadata().
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_embedded_json(text: str, opening: str):
    """Decode one complete JSON value without absorbing later prose/examples."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text or ""):
        if character != opening:
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("response contains no complete JSON value")


def generate_terms(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    match_script_order: bool = False,
    app_config=None,
) -> List[str]:
    video_script = utils.remove_pause_tags(video_script or "").strip()
    if match_script_order:
        goal = (
            f"Generate {amount} chronological stock-video search terms that follow "
            "the order of topics in the video script."
        )
        ordering_rule = (
            "6. keep the terms in the same order as the script narration; "
            "earlier terms must describe earlier visual moments."
        )
        # In sequential keyword mode, example counts must match amount to avoid misleading models with a fixed 4-example prompt,
        # which would cause long scripts to return too few keywords and compromise footage coverage.
        example_terms = [
            "opening visual topic",
            *[f"script visual topic {index}" for index in range(2, max(amount, 1))],
            "final visual topic",
        ]
        output_example = json.dumps(example_terms[:amount], ensure_ascii=False)
    else:
        goal = (
            f"Generate {amount} search terms for stock videos, depending on the "
            "subject of a video."
        )
        ordering_rule = ""
        output_example = (
            '["search term 1", "search term 2", "search term 3",'
            '"search term 4", "search term 5"]'
        )

    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
{goal}

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.
{ordering_rule}

## Output Example:
{output_example}

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}, match_script_order: {match_script_order}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        search_terms = []
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                # generate_terms has a public return type of List[str]. Returning raw provider error messages leads downstream
                # null-checks to misinterpret non-empty strings as valid keywords, causing download loops to iterate characters
                # and generate pointless requests. Return an empty list here so the orchestrator halts immediately at the true fault location.
                logger.error(f"failed to generate video terms: {response}")
                return []
            search_terms = json.loads(_strip_code_fence(response))
        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                try:
                    search_terms = _extract_embedded_json(response, "[")
                except ValueError as e:
                    logger.warning(f"failed to generate video terms: {str(e)}")

        # Apply the same contract to direct JSON and prose-wrapped recovery.
        # Otherwise a nonempty array of numbers or objects reaches material search.
        if not isinstance(search_terms, list) or not all(
            isinstance(term, str) for term in search_terms
        ):
            logger.error("response is not a list of strings.")
            search_terms = []

        # The model may ignore the requested count or return blank strings.
        # Enforce the prompt contract before material providers are contacted;
        # repeated topics remain meaningful in chronological mode.
        search_terms = [term.strip() for term in search_terms if term.strip()][:amount]
        if search_terms:
            break
        if i < _max_retries - 1:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


# =============================================================================
# Social publishing metadata
#
# Generate title, caption, and hashtags commonly used for publishing to short-form video platforms based on video topic and script.
# Reuses existing LLM providers without connecting to external publishing platforms or altering the primary video synthesis pipeline.
# =============================================================================

# Platforms vary in text length and hashtag preferences. Conservative upper bounds avoid downstream truncation needs.
SOCIAL_PLATFORMS = {
    "tiktok": {"title_max": 100, "caption_max": 2200, "hashtag_count": 5},
    "youtube_shorts": {"title_max": 100, "caption_max": 5000, "hashtag_count": 3},
    "instagram_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 8},
    "facebook_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 5},
}
DEFAULT_SOCIAL_PLATFORM = "tiktok"
DEFAULT_SOCIAL_LANGUAGE = "auto"
MAX_SOCIAL_SUBJECT_LENGTH = 500
MAX_SOCIAL_SCRIPT_LENGTH = 8000
MAX_SOCIAL_LANGUAGE_LENGTH = 64

SOCIAL_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube_shorts": "YouTube Shorts",
    "instagram_reels": "Instagram Reels",
    "facebook_reels": "Facebook Reels",
}

# Generic fallback hashtags when LLM is unavailable. Language-agnostic by design to ensure functional structures across locales.
DEFAULT_SOCIAL_HASHTAGS = [
    "#shorts",
    "#viral",
    "#trending",
    "#fyp",
    "#video",
    "#reels",
    "#creator",
    "#content",
]


def _resolve_social_platform(platform: str | None) -> str:
    value = (platform or "").strip().lower()
    return value if value in SOCIAL_PLATFORMS else DEFAULT_SOCIAL_PLATFORM


def _normalize_social_language(language: str | None) -> str:
    value = (language or DEFAULT_SOCIAL_LANGUAGE).strip()
    if len(value) > MAX_SOCIAL_LANGUAGE_LENGTH:
        logger.warning(
            "social metadata language is too long and will be truncated to "
            f"{MAX_SOCIAL_LANGUAGE_LENGTH} characters."
        )
        value = value[:MAX_SOCIAL_LANGUAGE_LENGTH]
    return value or DEFAULT_SOCIAL_LANGUAGE


def _limit_social_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API routes limit content length; guard internal or direct calls here to prevent excessive payloads and token costs.
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _social_language_instruction(language: str | None) -> str:
    language = _normalize_social_language(language)
    if language.lower() == DEFAULT_SOCIAL_LANGUAGE:
        return (
            "Use the same language as the video subject and script. If the subject "
            "and script use different languages, prefer the script language."
        )

    return f'Write "title" and "caption" in this language: {language}.'


def _clamp_text(text, max_length: int) -> str:
    value = ("" if text is None else str(text)).strip()
    if max_length and len(value) > max_length:
        return value[:max_length].rstrip()
    return value


def _normalize_hashtags(raw, count: int) -> List[str]:
    """
    Normalize hashtags returned by LLM into `#tag` format.

    LLMs may return strings, lists, spaced phrases, duplicates, or punctuation-laden tokens.
    Centralized sanitization ensures stable schema outputs, preventing empty, duplicate, or malformed tags on platforms.
    """
    if isinstance(raw, str):
        candidates = re.split(r"[\s,]+", raw)
    elif isinstance(raw, (list, tuple)):
        # Each list item is treated as an individual tag, so "du lich" becomes "#dulich" rather than two separate tags.
        candidates = [str(entry) for entry in raw]
    else:
        candidates = []

    seen = set()
    result: List[str] = []
    for item in candidates:
        # \w does not match combining marks, so keep them explicitly: they are
        # the vowel signs and tone marks of Devanagari, Bengali, Tamil and Thai
        # tags, and dropping them misspells the tag ("#भारत" became "#भरत").
        tag = "".join(
            character
            for character in item
            if character.isalnum()
            or character == "_"
            or unicodedata.category(character).startswith("M")
        )
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(f"#{tag}")
        if count and len(result) >= count:
            break
    return result


def build_social_metadata_prompt(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> str:
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    platform = _resolve_social_platform(platform)
    spec = SOCIAL_PLATFORMS[platform]
    label = SOCIAL_PLATFORM_LABELS.get(platform, platform)
    language_instruction = _social_language_instruction(language)

    prompt = f"""
# Role: Short-Video Social Media Copywriter

## Goal
Write engaging publishing metadata for a short video that will be posted on {label}.

## Constraints
1. Respond ONLY with a single valid minified JSON object. No markdown, no code fences, no commentary.
2. The JSON must contain exactly these keys: "title", "caption", "hashtags".
3. "title": a catchy hook, at most {spec["title_max"]} characters.
4. "caption": an engaging description that ends with a call to action, at most {spec["caption_max"]} characters. Do not put hashtags inside the caption.
5. "hashtags": a JSON array of exactly {spec["hashtag_count"]} strings. Each must start with "#", contain no spaces, and be relevant to the topic and to {label}.
6. {language_instruction}

## Output Example
{{"title":"...","caption":"...","hashtags":["#example","#video"]}}

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()
    return prompt


def _parse_social_metadata(response: str, platform: str) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]

    data = None
    try:
        data = json.loads(_strip_code_fence(response))
    except Exception:
        # Some models wrap JSON in explanatory text or markdown fences. Extract the first JSON object for stable parsing.
        data = _extract_embedded_json(response, "{")

    if not isinstance(data, dict):
        raise ValueError("social metadata response is not a JSON object")

    title = _clamp_text(data.get("title", ""), spec["title_max"])
    caption = _clamp_text(data.get("caption", ""), spec["caption_max"])
    hashtags = _normalize_hashtags(data.get("hashtags", []), spec["hashtag_count"])

    if not title and not caption:
        raise ValueError("social metadata response is missing both title and caption")

    return {"title": title, "caption": caption, "hashtags": hashtags}


def _fallback_social_metadata(
    video_subject: str, video_script: str, platform: str
) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]
    subject = (video_subject or "").strip()
    script = (video_script or "").strip()

    title = subject
    if not title and script:
        # Chinese/Japanese sentence endings typically lack spaces; English requires whitespace to avoid splitting decimals like 2.0.
        title = re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", script, maxsplit=1)[0]

    return {
        "title": _clamp_text(title, spec["title_max"]),
        "caption": _clamp_text(script or subject, spec["caption_max"]),
        "hashtags": _normalize_hashtags(DEFAULT_SOCIAL_HASHTAGS, spec["hashtag_count"]),
    }


def generate_social_metadata(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> dict:
    """
    Generate short-form video publishing metadata.

    Fixed return schema: `{"title": str, "caption": str, "hashtags": List[str]}`.
    Falls back to heuristic extraction if LLM is unavailable or responses are malformed,
    ensuring callers always receive a structured, editable payload.
    """
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    prompt = build_social_metadata_prompt(
        video_subject=video_subject,
        video_script=video_script,
        language=language,
        platform=platform,
    )
    logger.info(f"generating social metadata: platform={platform}, language={language}")

    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and response.startswith("Error: "):
                logger.error(f"failed to generate social metadata: {response}")
                break
            metadata = _parse_social_metadata(response, platform)
            logger.success(f"completed: \n{metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"failed to parse social metadata: {str(e)}")

        if i < _max_retries - 1:
            logger.warning(
                f"failed to generate social metadata, trying again... {i + 1}"
            )

    logger.warning("falling back to heuristic social metadata")
    return _fallback_social_metadata(video_subject, video_script, platform)


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
