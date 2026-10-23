"""Application implementation - ASGI."""

import math
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import config
from app.controllers import base
from app.models.exception import HttpException
from app.router import root_api_router
from app.utils import utils


@asynccontextmanager
async def application_lifespan(_: FastAPI):
    """Centrally handle API process startup recovery and shutdown logging."""
    logger.info("startup event")

    configured_api_key = config.app.get("api_key", "")
    if configured_api_key in (None, ""):
        logger.warning(
            "API key authentication is disabled; keep the API on a trusted network"
        )
    elif isinstance(configured_api_key, str):
        # Only log protection scope; never log the key, length, or hash to prevent credentials entering logs.
        logger.info("API key authentication is enabled for /api/v1 and /tasks")
    else:
        logger.error(
            "API key authentication is misconfigured: app.api_key must be a string"
        )

    # Cross-platform publishing runs in the current thread pool and is not resumed after restarts.
    # At startup, convert active states in Redis whose process is lost to failed so tasks can be deleted.
    from app.services import task as task_service

    task_service.recover_interrupted_cross_posts()

    # Redis queue entries persist across API restarts. No task_done callback
    # exists in the new process to dispatch them, so fill its worker slots now.
    from app.controllers.manager.redis_manager import RedisTaskManager
    from app.controllers.v1 import video as video_controller

    if isinstance(video_controller.task_manager, RedisTaskManager):
        video_controller.task_manager.resume_queued_tasks()
    try:
        yield
    finally:
        logger.info("shutdown event")


def exception_handler(request: Request, e: HttpException):
    return JSONResponse(
        status_code=e.status_code,
        content=utils.get_response(e.status_code, e.data, e.message),
    )


def validation_exception_handler(request: Request, e: RequestValidationError):
    # Rejected inputs can contain NaN/Infinity, and custom validators attach
    # exception objects to ctx. Neither can be emitted by JSONResponse directly.
    errors = jsonable_encoder(
        e.errors(),
        custom_encoder={
            float: lambda value: value if math.isfinite(value) else str(value),
            Exception: str,
        },
    )
    # Describe the first rejected field instead of always claiming a missing
    # field: out-of-range values and malformed JSON are not "field required".
    message = "invalid request parameters"
    if errors and isinstance(errors[0], dict):
        location = ".".join(str(part) for part in errors[0].get("loc") or ())
        detail = errors[0].get("msg") or message
        message = f"{location}: {detail}" if location else str(detail)
    return JSONResponse(
        status_code=400,
        content=utils.get_response(status=400, data=errors, message=message),
    )


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _normalize_allowed_origin(raw_origin: str) -> str | None:
    """Fold configuration entries into the canonical Origin format sent by browsers.

    RFC 6454 defines Origin headers as ``scheme://host[:port]``: no path, no trailing
    slash, normalized lowercase scheme/host, with default protocol ports omitted. Values copied
    from browser address bars like ``https://frontend.example/`` or explicit default ports like
    ``https://frontend.example:443`` do not match ``https://frontend.example`` character-by-character,
    causing whitelists to silently fail.

    Returns ``None`` if the string cannot match any valid Origin (missing scheme, scheme not http/https,
    empty/whitespace hostname, malformed IPv6, or invalid port). The caller discards and logs a warning.
    Must return ``None`` on error rather than raising ValueError so the API module import never crashes.
    """

    if raw_origin == "*":
        return raw_origin

    try:
        parsed = urlsplit(raw_origin)
        scheme = parsed.scheme.lower()
        # Netloc validation occurs when accessing hostname/port: malformed IPv6 is raised
        # by urlsplit(), while non-numeric or out-of-range ports are raised when accessing port.
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None

    # Only http/https carry browser Origin semantics with omittable default ports.
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    if any(character.isspace() for character in host):
        return None

    # Explicitly lowercase without relying on urlsplit casing behavior across Python versions.
    host = host.lower()
    # Browsers retain brackets for IPv6 literals; default ports like ``:443``, ``:0443``,
    # or ``:80`` must fold into port-free representations to match real frontends.
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def parse_cors_allowed_origins(raw_origins: str | None) -> list[str]:
    """Parse allowed origins whitelist for browser CORS.

    CORS only restricts in-browser cross-origin JavaScript; curl, Postman, n8n,
    and server SDKs are unaffected. Returns an empty list when unconfigured (same-origin by default).
    Users with separate web frontends can enable access via ``CORS_ALLOWED_ORIGINS``.

    Each origin is folded into canonical form so ``https://a.example/``, ``HTTPS://A.Example``,
    and ``https://a.example:443`` yield identical matches. Invalid entries are discarded with warnings
    without interrupting parsing or crashing the API at import time.
    """

    if not raw_origins:
        return []

    # Strip surrounding whitespace and ignore empty entries to tolerate trailing commas
    # such as ``https://a.example, https://b.example,`` without producing unmatchable entries.
    origins: list[str] = []
    for candidate in raw_origins.split(","):
        item = candidate.strip()
        if not item:
            continue
        origin = _normalize_allowed_origin(item)
        if origin is None:
            logger.warning(
                f"ignoring configured CORS origin that cannot match a browser "
                f"Origin header: {item!r}"
            )
            continue
        if origin not in origins:
            origins.append(origin)
    return origins


def configure_cors(instance: FastAPI, allowed_origins: list[str]) -> None:
    """Configure CORS according to explicit whitelist; empty whitelist preserves default same-origin policy."""

    if not allowed_origins:
        logger.info(
            "browser cross-origin API access is disabled; set "
            "CORS_ALLOWED_ORIGINS to enable trusted origins"
        )
        return

    allow_all_origins = "*" in allowed_origins
    configured_api_key = config.app.get("api_key", "")
    if allow_all_origins and configured_api_key in (None, ""):
        # ``*`` is an explicit user opt-in for compatibility, so startup is allowed;
        # however, in unauthenticated mode it permits any webpage to call the API, so log a security warning.
        logger.warning(
            "CORS allows every browser origin while API key authentication is "
            "disabled; configure app.api_key or restrict CORS_ALLOWED_ORIGINS"
        )

    instance.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        # Starlette reflects arbitrary origins when ``*`` and credentials are both enabled.
        # Wildcard mode does not require cookies, so disable credentials; explicit origins
        # retain credential support to avoid breaking existing frontend clients.
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        # Modern browsers send Private Network Access preflights when remote HTTPS accesses
        # local/LAN APIs. Only exact whitelisted origins are permitted; wildcard mode rejects PNA
        # to prevent malicious websites from port-scanning private network services.
        allow_private_network=not allow_all_origins,
    )


def is_browser_origin_allowed(
    request: Request, allowed_origins: list[str]
) -> bool:
    """Check whether the browser request Origin is same-origin or in the explicit whitelist."""

    origin = request.headers.get("origin")
    if not origin:
        # curl, Postman, n8n, and SDKs usually omit the Origin header. Allow these requests
        # to avoid breaking existing non-browser API client integrations.
        return True
    if "*" in allowed_origins or origin in allowed_origins:
        return True

    # Browsers may also send Origin on same-origin POST. Compare scheme + authority only;
    # reverse-proxy deployments missing forwarded headers can configure CORS_ALLOWED_ORIGINS explicitly.
    request_url = urlsplit(str(request.url))
    request_origin = f"{request_url.scheme}://{request_url.netloc}"
    return origin == request_origin


def configure_browser_access(instance: FastAPI, allowed_origins: list[str]) -> None:
    """Configure both server-side Origin verification and browser CORS response policies."""

    @instance.middleware("http")
    async def reject_untrusted_browser_origin(request: Request, call_next):
        """Reject untrusted browser origins, covering simple requests that bypass CORS preflight."""

        if not is_browser_origin_allowed(request, allowed_origins):
            origin = request.headers.get("origin", "")
            logger.warning(
                f"blocked untrusted browser origin: method={request.method}, "
                f"path={request.url.path}, origin={origin}"
            )
            return JSONResponse(
                status_code=403,
                content=utils.get_response(
                    status=403,
                    message="cross-origin browser request is not allowed",
                ),
            )

        return await call_next(request)

    # Registered last, CORS middleware wraps Origin protection: trusted preflights succeed,
    # untrusted preflights are rejected, and actual requests enter the 403 validation above.
    configure_cors(instance, allowed_origins)


def get_application() -> FastAPI:
    """Initialize FastAPI application.

    Returns:
       FastAPI: Application object instance.

    """
    instance = FastAPI(
        title=config.project_name,
        description=config.project_description,
        version=config.project_version,
        debug=False,
        lifespan=application_lifespan,
    )
    instance.include_router(root_api_router)
    instance.add_exception_handler(HttpException, exception_handler)
    instance.add_exception_handler(RequestValidationError, validation_exception_handler)
    return instance


app = get_application()


@app.middleware("http")
async def protect_generated_task_files(request: Request, call_next):
    """Protect task artifact static routes from unauthenticated downloads.

    ``/tasks`` is mounted via StaticFiles and cannot reuse APIRouter dependencies directly,
    so verify_token is invoked in middleware. Authorization is bypassed when api_key is unset;
    OPTIONS preflights pass through to the CORS middleware.
    """

    request_path = request.url.path
    is_task_file = request_path == "/tasks" or request_path.startswith("/tasks/")
    if is_task_file and request.method != "OPTIONS":
        try:
            base.verify_token(request)
        except HttpException as exception:
            return exception_handler(request, exception)

    return await call_next(request)


# Follow browser same-origin policy by default; allow cross-origin only when explicitly configured.
cors_allowed_origins = parse_cors_allowed_origins(
    os.getenv("CORS_ALLOWED_ORIGINS", "")
)
configure_browser_access(app, cors_allowed_origins)

task_dir = utils.task_dir()
app.mount("/tasks", StaticFiles(directory=task_dir, html=True), name="")

public_dir = utils.public_dir()
app.mount("/", StaticFiles(directory=public_dir, html=True), name="")
