import secrets
from typing import Annotated
from uuid import uuid4

from fastapi import Header, Request

from app.config import config
from app.models.exception import HttpException

MAX_TASK_ID_LENGTH = 128


def normalize_task_id(value: object) -> str:
    """Return a log-safe request ID, replacing invalid client input with a UUID."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_TASK_ID_LENGTH
        or not value.isprintable()
    ):
        return str(uuid4())
    return value


def get_task_id(request: Request) -> str:
    return normalize_task_id(request.headers.get("x-task-id"))


def get_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    return api_key


def get_api_key_values(request: Request) -> list[str]:
    """Return all API Key headers in the request, preserving duplicates for security checks."""

    # Starlette Headers provides getlist() to distinguish duplicate headers from proxies or clients.
    # Lightweight Request test doubles use standard dicts, so keep a compatible fallback.
    get_list = getattr(request.headers, "getlist", None)
    if callable(get_list):
        return [value for value in get_list("x-api-key") if isinstance(value, str)]

    api_key = get_api_key(request)
    return [api_key] if isinstance(api_key, str) else []


def verify_token(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
):
    """Verify API Key based on configuration.

    An empty key preserves existing unauthenticated local access. When a non-empty key
    is explicitly configured, API routes and task artifact downloads require clients to provide
    the matching value in the ``x-api-key`` header. Declaring the parameter also enables Swagger UI testing.
    """

    configured_key = config.app.get("api_key", "")
    if configured_key in (None, ""):
        return None

    # Config item must be a string. Reject lists, numbers, etc. to avoid subtle auth issues;
    # error messages never include the actual key.
    if not isinstance(configured_key, str):
        raise HttpException(
            task_id=get_task_id(request),
            status_code=500,
            message="API authentication is misconfigured",
        )

    # FastAPI parameter declares x-api-key in OpenAPI; validation reads Request directly
    # to detect duplicate headers. Because clients and reverse proxies handle duplicate header ordering
    # inconsistently, duplicates must be rejected rather than implicitly taking first or last.
    token_values = get_api_key_values(request)
    if not token_values and isinstance(x_api_key, str):
        token_values = [x_api_key]

    if len(token_values) != 1:
        raise HttpException(
            task_id=get_task_id(request),
            status_code=401,
            message="invalid API key",
        )

    # compare_digest on str only supports ASCII. Headers are untrusted input, where non-ASCII
    # could trigger TypeError. Encoding to UTF-8 bytes preserves constant-time comparison and supports Unicode keys.
    token = token_values[0]
    if not secrets.compare_digest(
        token.encode("utf-8"), configured_key.encode("utf-8")
    ):
        raise HttpException(
            task_id=get_task_id(request),
            status_code=401,
            message="invalid API key",
        )

    return None
